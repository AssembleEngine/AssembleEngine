//! Aiming in the world (feature `aim`, M14): a unit's [`Facing`] (its
//! frame's orientation, synced), where its bones are in the world
//! ([`Anim::bone_world`]), and its [`Weapons`] (aim-from and fire-from
//! bones, aim requests and answers). Opt-in, as `nav`: every component
//! compiled takes an entity id in the module, used or not.

use crate::anim::{Anim, Axis, norm};
use crate::nav::Body;
use crate::{Component, abi};

/// A unit's orientation: the rotation from its frame (x forward, y up, z
/// right: the model's) to the world's (nav x, up, nav y; the renderer's x,
/// y, z), a unit quaternion (x, y, z, w). Synced and saved; whoever places
/// the unit writes it (the game; later the movement service). Tilt is
/// allowed. Identity: facing nav +x, upright.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0012)]
pub struct Facing {
    #[field(view = "los")]
    pub rot: [f32; 4],
}

const _: () = assert!(<Facing as Component>::ID == abi::engine::FACING);

impl Default for Facing {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Facing {
    pub const IDENTITY: Facing = Facing {
        rot: [0.0, 0.0, 0.0, 1.0],
    };

    /// Upright, facing ground direction `(dx, dy)` (nav; need not be unit
    /// length; zero: [`Facing::IDENTITY`]).
    pub fn along(dx: f32, dy: f32) -> Facing {
        let len = libm::sqrtf(dx * dx + dy * dy);
        if len == 0.0 {
            return Self::IDENTITY;
        }
        // The shortest turn from +x to (dx, 0, dy), about up.
        let (c, s) = (dx / len, dy / len);
        if c <= -1.0 + 1e-6 {
            return Facing {
                rot: [0.0, 1.0, 0.0, 0.0],
            };
        }
        // Half-way vector: (1 + c, 0, s) normalised; the axis +x × it.
        let q = [0.0, -s, 0.0, 1.0 + c];
        let n = libm::sqrtf(q[1] * q[1] + q[3] * q[3]);
        Facing {
            rot: [0.0, q[1] / n, 0.0, q[3] / n],
        }
    }

    /// Turned further by `q` (in the unit's frame: a tilt about
    /// [`Axis::RIGHT`] is `rot · q`).
    pub fn then(self, q: [f32; 4]) -> Facing {
        Facing {
            rot: qnorm(qmul(self.rot, q)),
        }
    }

    /// A world direction in the unit's frame (by its full orientation).
    pub fn to_unit(&self, dir: [f32; 3]) -> [f32; 3] {
        qrot(qconj(self.rot), dir)
    }

    /// A direction in the unit's frame in the world.
    pub fn to_world(&self, dir: [f32; 3]) -> [f32; 3] {
        qrot(self.rot, dir)
    }
}

/// The rotation by `angle` radians about `axis` (a unit vector), as
/// [`Facing::then`] takes it.
pub fn rotation(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let a = norm(axis);
    let (s, c) = (libm::sinf(angle * 0.5), libm::cosf(angle * 0.5));
    [a[0] * s, a[1] * s, a[2] * s, c]
}

/// Weapons per unit in [`Weapons`].
pub const WEAPONS: usize = 4;

/// A unit's weapons as aiming sees them (the weapons system is the game's,
/// for now): per weapon, the bone its aim is measured from and the bone
/// shots leave from (the script writes them, and may alternate barrels by
/// changing `fire_from` after each shot), and its aim requests. Whoever
/// asks a weapon to aim ([`Weapons::ask`]) posts the `AIM` callin; the
/// script turns its bones and answers ([`Weapons::set_aimed`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Component)]
#[component(id = 0xAE00_0000_0000_0013)]
pub struct Weapons {
    pub aim_from: [u16; WEAPONS],
    pub fire_from: [u16; WEAPONS],
    /// The world direction each weapon is asked to point (unit length),
    /// three floats a weapon.
    pub dir: [f32; 3 * WEAPONS],
    /// Requests made, by weapon (one more with each new direction).
    pub asked: [u32; WEAPONS],
    /// The request the script last finished aiming for.
    pub aimed: [u32; WEAPONS],
}

const _: () = assert!(<Weapons as Component>::ID == abi::engine::WEAPONS);

impl Weapons {
    /// Ask weapon `w` to point along world direction `dir` (normalised
    /// here). Post `callin::AIM` with it.
    pub fn ask(&mut self, w: usize, dir: [f32; 3]) {
        let d = norm(dir);
        self.dir[3 * w..3 * w + 3].copy_from_slice(&d);
        self.asked[w] = self.asked[w].wrapping_add(1);
    }

    /// The direction weapon `w` was last asked to point.
    pub fn dir(&self, w: usize) -> [f32; 3] {
        [self.dir[3 * w], self.dir[3 * w + 1], self.dir[3 * w + 2]]
    }

    /// Weapon `w` has a request not yet answered.
    pub fn wants_aim(&self, w: usize) -> bool {
        self.asked[w] != self.aimed[w]
    }

    /// The script answers request `request` of weapon `w` (the `asked` it
    /// aimed for): aimed, unless a newer request came meanwhile.
    pub fn set_aimed(&mut self, w: usize, request: u32) {
        self.aimed[w] = request;
    }

    /// Weapon `w` points where it was last asked (its turns done): it may
    /// fire.
    pub fn is_aimed(&self, w: usize) -> bool {
        self.asked[w] != 0 && self.aimed[w] == self.asked[w]
    }
}

/// Where a bone is in the world ([`Anim::bone_world`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoneWorld {
    /// Its origin: nav x, height above the unit's ground point, nav y.
    pub pos: [f32; 3],
    /// The rotation from the unit's frame at rest to the world, as the
    /// bone carries it.
    pub rot: [f32; 4],
}

impl BoneWorld {
    /// Where `axis` (in the unit's frame, as the bone was at rest: its
    /// pointing, e.g. [`Axis::FORWARD`]) points now, in the world.
    pub fn dir(&self, axis: Axis) -> [f32; 3] {
        assert!(!axis.bone, "BoneWorld::dir takes an axis in the unit's frame");
        qrot(self.rot, norm(axis.v))
    }
}

impl Anim {
    /// Where `bone` is in the world for a unit at `body` facing `facing`,
    /// as these channels pose it (commands not yet taken by the engine do
    /// not count, as for [`Anim::bounds`]): the model fitted to the unit's
    /// radius as the renderer draws it, from the unit's ground point (the
    /// sim has no terrain height: heights are above that point). Where
    /// aim directions are measured from, and shots leave from.
    pub fn bone_world(&self, bone: u16, body: &Body, facing: &Facing) -> BoneWorld {
        const N: usize = size_of::<Anim>();
        let mut input = [0u8; N + 32];
        // SAFETY: `Anim` is plain data (`repr(C)`, no padding read back).
        input[..N].copy_from_slice(unsafe { core::slice::from_raw_parts((self as *const Anim).cast::<u8>(), N) });
        let words = [
            bone as u32,
            body.x.to_bits(),
            body.y.to_bits(),
            body.radius.to_bits(),
            facing.rot[0].to_bits(),
            facing.rot[1].to_bits(),
            facing.rot[2].to_bits(),
            facing.rot[3].to_bits(),
        ];
        for (k, w) in words.iter().enumerate() {
            input[N + 4 * k..N + 4 * k + 4].copy_from_slice(&w.to_le_bytes());
        }
        let mut out = [0f32; 7];
        let op = abi::name_id(abi::query::ANIM_BONE_WORLD);
        let n = unsafe {
            crate::host::ae_query(
                op,
                input.as_ptr() as usize as u32,
                input.len() as u32,
                out.as_mut_ptr() as usize as u32,
                28,
            )
        };
        assert!(n == 28, "the engine has no animation service (`ae_query` {})", n);
        BoneWorld {
            pos: [out[0], out[1], out[2]],
            rot: [out[3], out[4], out[5], out[6]],
        }
    }
}

fn qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

fn qconj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

fn qnorm(q: [f32; 4]) -> [f32; 4] {
    let n = libm::sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if n > 0.0 {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// `v` rotated by `q` (as `ae_model::anim::qrot`).
fn qrot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let r = qmul(qmul(q, [v[0], v[1], v[2], 0.0]), qconj(q));
    [r[0], r[1], r[2]]
}
