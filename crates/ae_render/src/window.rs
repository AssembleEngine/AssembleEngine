//! SDL3 window and event pump (main thread only), and the Vulkan surface
//! factory behind `SurfaceSource` (§3.6). SDL never owns the Vulkan device or
//! swapchain; the render thread does. Resize, minimize and close are
//! reported as events for the caller to forward to the render thread.

use anyhow::{Result, anyhow, bail};
use ash::vk;
use sdl3_sys::everything as sdl;
use std::ffi::{CStr, CString, c_char};

/// Creates the presentation surface for an instance. Implemented by the
/// windowing layer, so the decision for SDL3 stays reversible.
pub trait SurfaceSource {
    /// Instance extensions the surface needs.
    fn instance_extensions(&self) -> Result<Vec<*const c_char>>;
    fn create_surface(&self, entry: &ash::Entry, instance: &ash::Instance) -> Result<vk::SurfaceKHR>;
}

fn sdl_error() -> String {
    unsafe {
        let error = sdl::SDL_GetError();
        if error.is_null() {
            "unknown SDL error".to_string()
        } else {
            CStr::from_ptr(error).to_string_lossy().into_owned()
        }
    }
}

/// What the event pump saw since the last call.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindowEvents {
    pub quit: bool,
    /// The drawable size changed (pixels).
    pub resized: Option<(u32, u32)>,
    pub minimized: bool,
    /// Visible again (restored, shown, exposed, resized, focused), after
    /// the last minimize in this batch.
    pub restored: bool,
    /// Mouse wheel steps (positive: away from the user).
    pub wheel: f32,
    /// Mouse motion in pixels while the right or middle button is held.
    pub drag: (f32, f32),
    /// Keys pressed in this batch (`keys::*` bits).
    pub pressed: u32,
}

/// Toggle keys reported in [`WindowEvents::pressed`].
pub mod keys {
    pub const T: u32 = 1 << 0;
    pub const L: u32 = 1 << 1;
    pub const G: u32 = 1 << 2;
    pub const C: u32 = 1 << 3;
    pub const PLUS: u32 = 1 << 4;
    pub const MINUS: u32 = 1 << 5;
    pub const P: u32 = 1 << 6;
    pub const V: u32 = 1 << 7;
    pub const M: u32 = 1 << 8;
    pub const B: u32 = 1 << 9;
}

/// Movement keys currently held (WASD or arrows pan, Q/E orbit, R/F tilt,
/// shift faster).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeldKeys {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub orbit_left: bool,
    pub orbit_right: bool,
    pub tilt_up: bool,
    pub tilt_down: bool,
    pub fast: bool,
}

/// The SDL video subsystem plus one Vulkan-capable window. Not `Send`: SDL
/// requires window creation and event pumping on the main thread.
pub struct SdlWindow {
    window: *mut sdl::SDL_Window,
    /// Print window events (AE_RENDER_TRACE_EVENTS=1).
    trace: bool,
    _not_send: std::marker::PhantomData<*mut ()>,
}

impl SdlWindow {
    pub fn new(title: &str, width: u32, height: u32) -> Result<Self> {
        unsafe {
            if !sdl::SDL_Init(sdl::SDL_INIT_VIDEO) {
                bail!("SDL_Init failed: {}", sdl_error());
            }
            let title = CString::new(title)?;
            let window = sdl::SDL_CreateWindow(
                title.as_ptr(),
                width as i32,
                height as i32,
                sdl::SDL_WINDOW_VULKAN | sdl::SDL_WINDOW_RESIZABLE,
            );
            if window.is_null() {
                let error = sdl_error();
                sdl::SDL_Quit();
                bail!("SDL_CreateWindow failed: {error}");
            }
            Ok(Self {
                window,
                trace: std::env::var("AE_RENDER_TRACE_EVENTS").is_ok_and(|raw| raw.trim() == "1"),
                _not_send: std::marker::PhantomData,
            })
        }
    }

    pub fn pixel_size(&self) -> (u32, u32) {
        let (mut width, mut height) = (0, 0);
        unsafe { sdl::SDL_GetWindowSizeInPixels(self.window, &mut width, &mut height) };
        (width.max(0) as u32, height.max(0) as u32)
    }

    /// Request a window size (logical units). Used by the exercise mode.
    pub fn set_size(&mut self, width: u32, height: u32) {
        unsafe { sdl::SDL_SetWindowSize(self.window, width as i32, height as i32) };
    }

    pub fn set_title(&mut self, title: &str) {
        if let Ok(title) = CString::new(title) {
            unsafe { sdl::SDL_SetWindowTitle(self.window, title.as_ptr()) };
        }
    }

    pub fn minimize(&mut self) {
        unsafe { sdl::SDL_MinimizeWindow(self.window) };
    }

    pub fn restore(&mut self) {
        unsafe { sdl::SDL_RestoreWindow(self.window) };
    }

    /// Movement keys held now.
    pub fn held(&self) -> HeldKeys {
        let mut count = 0;
        let state = unsafe { sdl::SDL_GetKeyboardState(&mut count) };
        if state.is_null() {
            return HeldKeys::default();
        }
        let state = unsafe { std::slice::from_raw_parts(state, count.max(0) as usize) };
        let down = |code: sdl::SDL_Scancode| state.get(code.0 as usize).copied().unwrap_or(false);
        HeldKeys {
            forward: down(sdl::SDL_SCANCODE_W) || down(sdl::SDL_SCANCODE_UP),
            back: down(sdl::SDL_SCANCODE_S) || down(sdl::SDL_SCANCODE_DOWN),
            left: down(sdl::SDL_SCANCODE_A) || down(sdl::SDL_SCANCODE_LEFT),
            right: down(sdl::SDL_SCANCODE_D) || down(sdl::SDL_SCANCODE_RIGHT),
            orbit_left: down(sdl::SDL_SCANCODE_Q),
            orbit_right: down(sdl::SDL_SCANCODE_E),
            tilt_up: down(sdl::SDL_SCANCODE_R),
            tilt_down: down(sdl::SDL_SCANCODE_F),
            fast: down(sdl::SDL_SCANCODE_LSHIFT) || down(sdl::SDL_SCANCODE_RSHIFT),
        }
    }

    /// Drain pending events without blocking.
    pub fn pump(&mut self) -> WindowEvents {
        let mut events = WindowEvents::default();
        let mut event = std::mem::MaybeUninit::<sdl::SDL_Event>::zeroed();
        unsafe {
            while sdl::SDL_PollEvent(event.as_mut_ptr()) {
                let event = event.assume_init_ref();
                let kind = sdl::SDL_EventType(event.r#type);
                if self.trace && (0x200..0x300).contains(&event.r#type) {
                    eprintln!("[window] event {:#x} data {} {}", event.r#type, event.window.data1, event.window.data2);
                }
                let escape = kind == sdl::SDL_EVENT_KEY_DOWN && event.key.key == sdl::SDLK_ESCAPE;
                if kind == sdl::SDL_EVENT_QUIT || kind == sdl::SDL_EVENT_WINDOW_CLOSE_REQUESTED || escape {
                    events.quit = true;
                } else if kind == sdl::SDL_EVENT_KEY_DOWN && !event.key.repeat {
                    let key = event.key.key;
                    events.pressed |= match key {
                        sdl::SDLK_T => keys::T,
                        sdl::SDLK_L => keys::L,
                        sdl::SDLK_G => keys::G,
                        sdl::SDLK_C => keys::C,
                        sdl::SDLK_P => keys::P,
                        sdl::SDLK_V => keys::V,
                        sdl::SDLK_M => keys::M,
                        sdl::SDLK_B => keys::B,
                        sdl::SDLK_EQUALS | sdl::SDLK_PLUS | sdl::SDLK_KP_PLUS => keys::PLUS,
                        sdl::SDLK_MINUS | sdl::SDLK_KP_MINUS => keys::MINUS,
                        _ => 0,
                    };
                } else if kind == sdl::SDL_EVENT_MOUSE_WHEEL {
                    events.wheel += event.wheel.y;
                } else if kind == sdl::SDL_EVENT_MOUSE_MOTION {
                    if event.motion.state & (sdl::SDL_BUTTON_RMASK | sdl::SDL_BUTTON_MMASK) != 0 {
                        events.drag.0 += event.motion.xrel;
                        events.drag.1 += event.motion.yrel;
                    }
                } else if kind == sdl::SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED {
                    let (width, height) = (event.window.data1.max(0) as u32, event.window.data2.max(0) as u32);
                    events.resized = Some((width, height));
                    events.restored = true;
                } else if kind == sdl::SDL_EVENT_WINDOW_MINIMIZED {
                    events.minimized = true;
                    events.restored = false;
                } else if kind == sdl::SDL_EVENT_WINDOW_RESTORED
                    || kind == sdl::SDL_EVENT_WINDOW_SHOWN
                    || kind == sdl::SDL_EVENT_WINDOW_EXPOSED
                    || kind == sdl::SDL_EVENT_WINDOW_MAXIMIZED
                    || kind == sdl::SDL_EVENT_WINDOW_FOCUS_GAINED
                {
                    // Wayland never reports RESTORED after a minimize; any
                    // sign that the window is visible again counts.
                    events.restored = true;
                }
            }
        }
        events
    }
}

impl SurfaceSource for SdlWindow {
    fn instance_extensions(&self) -> Result<Vec<*const c_char>> {
        let mut count = 0u32;
        let names = unsafe { sdl::SDL_Vulkan_GetInstanceExtensions(&mut count) };
        if names.is_null() {
            bail!("SDL_Vulkan_GetInstanceExtensions failed: {}", sdl_error());
        }
        // SDL owns the array for the lifetime of the video subsystem.
        Ok(unsafe { std::slice::from_raw_parts(names, count as usize) }.to_vec())
    }

    fn create_surface(&self, _entry: &ash::Entry, instance: &ash::Instance) -> Result<vk::SurfaceKHR> {
        let mut surface = vk::SurfaceKHR::null();
        let ok = unsafe {
            sdl::SDL_Vulkan_CreateSurface(self.window, instance.handle(), std::ptr::null(), &mut surface)
        };
        if !ok {
            return Err(anyhow!("SDL_Vulkan_CreateSurface failed: {}", sdl_error()));
        }
        Ok(surface)
    }
}

impl Drop for SdlWindow {
    fn drop(&mut self) {
        unsafe {
            sdl::SDL_DestroyWindow(self.window);
            sdl::SDL_Quit();
        }
    }
}
