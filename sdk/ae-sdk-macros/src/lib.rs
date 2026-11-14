//! Macros of the sim module SDK. Each one writes a record into the
//! `ae.schema` custom section (layout in `ae_abi`). Lengths and names are
//! known here; sizes, offsets and component ids need the compiler, so the
//! record is a byte template patched in a `const` block.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as Tokens};
use quote::{format_ident, quote};
use syn::punctuated::Punctuated;
use syn::{Data, DeriveInput, Error, Fields, Ident, ItemFn, LitInt, Path, Token, parenthesized, parse_macro_input};

use ae_abi as abi;

/// Record bytes plus statements that patch type-dependent values into `__b`.
struct Record {
    bytes: Vec<u8>,
    patches: Vec<Tokens>,
}

impl Record {
    fn new(kind: u32) -> Self {
        let mut r = Record { bytes: Vec::new(), patches: Vec::new() };
        r.u32(kind);
        r.u32(0); // length, fixed in finish()
        r
    }
    fn u32(&mut self, v: u32) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes.extend_from_slice(&v.to_le_bytes());
    }
    fn raw_name(&mut self, s: &str) {
        self.bytes.extend_from_slice(s.as_bytes());
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
    }
    fn string(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.raw_name(s);
    }
    fn patch_u32(&mut self, value: Tokens) {
        let at = self.bytes.len();
        self.u32(0);
        self.patches.push(quote! { ::ae_sdk::abi::put_u32(&mut __b, #at, (#value) as u32); });
    }
    fn patch_u64(&mut self, value: Tokens) {
        let at = self.bytes.len();
        self.u64(0);
        self.patches.push(quote! { ::ae_sdk::abi::put_u64(&mut __b, #at, (#value) as u64); });
    }
    /// A `const` expression of type `[u8; N]`.
    fn finish(mut self) -> (usize, Tokens) {
        let len = self.bytes.len() as u32;
        self.bytes[4..8].copy_from_slice(&len.to_le_bytes());
        let n = self.bytes.len();
        let bytes = &self.bytes;
        let patches = &self.patches;
        (n, quote! { { let mut __b: [u8; #n] = [#(#bytes),*]; #(#patches)* __b } })
    }
}

/// The section static. Only wasm builds get the link section; native builds
/// (SDK unit tests) keep the bytes as a plain const.
fn section_static(ident: &Ident, n: usize, value: Tokens) -> Tokens {
    quote! {
        #[used]
        #[allow(non_upper_case_globals)]
        #[cfg_attr(target_family = "wasm", unsafe(link_section = "ae.schema"))]
        static #ident: [u8; #n] = #value;
    }
}

fn parse_id(meta: &syn::meta::ParseNestedMeta) -> syn::Result<u64> {
    let lit: LitInt = meta.value()?.parse()?;
    lit.base10_parse()
}

/// `#[derive(Component)]` on a `#[repr(C)] #[derive(Clone, Copy)]` struct
/// with named fields.
///
/// * `#[component(id = N)]`: stable id (default: FNV-1a of the type name).
/// * `#[component(event)]`: the type is an event stream (implements
///   `ae_sdk::Event` instead of `ae_sdk::Component`).
/// * `#[field(id = N)]`: stable field id (default: FNV-1a of the field name).
/// * `#[field(unhashed)]`, `#[field(unsynced)]`, `#[field(unsaved)]`: clear
///   the default flags.
/// * `#[field(bytes)]`: opaque bytes instead of a typed value.
/// * `#[field(view = "los")]`: the field's view class (`never`, the
///   default, `always`, `owner`, `los`, `radar`, `ghost`): what observers
///   may see of it through the copy layer.
#[proc_macro_derive(Component, attributes(component, field))]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    derive(input).unwrap_or_else(Error::into_compile_error).into()
}

fn derive(input: DeriveInput) -> syn::Result<Tokens> {
    let ty = &input.ident;
    if !input.generics.params.is_empty() {
        return Err(Error::new_spanned(&input.generics, "components cannot be generic"));
    }
    let mut repr_c = false;
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("repr")) {
        attr.parse_nested_meta(|m| {
            if m.path.is_ident("C") || m.path.is_ident("transparent") {
                repr_c = true;
            }
            Ok(())
        })?;
    }
    if !repr_c {
        return Err(Error::new_spanned(ty, "components need #[repr(C)]: the host and other modules read their layout"));
    }
    let name = ty.to_string();
    let mut id = abi::name_id(&name);
    let mut event = false;
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("component")) {
        attr.parse_nested_meta(|m| {
            if m.path.is_ident("id") {
                id = parse_id(&m)?;
            } else if m.path.is_ident("event") {
                event = true;
            } else {
                return Err(m.error("expected `id = N` or `event`"));
            }
            Ok(())
        })?;
    }
    let Data::Struct(data) = &input.data else { return Err(Error::new_spanned(ty, "components must be structs")) };
    let Fields::Named(fields) = &data.fields else { return Err(Error::new_spanned(ty, "components need named fields")) };
    if fields.named.is_empty() {
        return Err(Error::new_spanned(ty, "components need at least one field (tags are not supported yet)"));
    }

    let mut r = Record::new(abi::KIND_COMPONENT);
    r.u64(id);
    r.patch_u32(quote! { ::core::mem::size_of::<#ty>() });
    r.patch_u32(quote! { ::core::mem::align_of::<#ty>() });
    r.u32(if event { abi::COMPONENT_EVENT } else { 0 });
    r.u32(fields.named.len() as u32);
    r.u32(name.len() as u32);
    r.u32(0);
    r.raw_name(&name);
    let mut field_ids = Vec::new();
    for field in &fields.named {
        let fname = field.ident.as_ref().unwrap();
        let fty = &field.ty;
        let fname_s = fname.to_string();
        let mut fid = abi::name_id(&fname_s);
        let mut flags = abi::FIELD_DEFAULT;
        let mut bytes = false;
        for attr in field.attrs.iter().filter(|a| a.path().is_ident("field")) {
            attr.parse_nested_meta(|m| {
                if m.path.is_ident("id") {
                    fid = parse_id(&m)?;
                } else if m.path.is_ident("unhashed") {
                    flags &= !abi::FIELD_HASHED;
                } else if m.path.is_ident("unsynced") {
                    flags &= !abi::FIELD_SYNCED;
                } else if m.path.is_ident("unsaved") {
                    flags &= !abi::FIELD_SAVED;
                } else if m.path.is_ident("bytes") {
                    bytes = true;
                } else if m.path.is_ident("view") {
                    let lit: syn::LitStr = m.value()?.parse()?;
                    let class = abi::view_class(&lit.value()).ok_or_else(|| m.error("view classes: never, always, owner, los, radar, ghost"))?;
                    flags = (flags & !abi::FIELD_VIEW_MASK) | class << abi::FIELD_VIEW_SHIFT;
                } else {
                    return Err(m.error("expected `id = N`, `unhashed`, `unsynced`, `unsaved`, `bytes` or `view = \"...\"`"));
                }
                Ok(())
            })?;
        }
        if field_ids.contains(&fid) {
            return Err(Error::new_spanned(fname, "two fields share a stable id"));
        }
        field_ids.push(fid);
        r.u64(fid);
        r.patch_u32(quote! { ::core::mem::offset_of!(#ty, #fname) });
        r.patch_u32(quote! { ::core::mem::size_of::<#fty>() });
        if bytes {
            r.u32(abi::FIELD_BYTES);
        } else {
            r.patch_u32(quote! { <#fty as ::ae_sdk::Field>::KIND });
        }
        r.u32(flags);
        r.u32(fname_s.len() as u32);
        r.u32(0);
        r.raw_name(&fname_s);
    }
    let (n, value) = r.finish();
    let trait_path = if event { quote!(::ae_sdk::Event) } else { quote!(::ae_sdk::Component) };
    let section = section_static(&format_ident!("__AE_SCHEMA_{}", ty), n, quote!(<#ty>::__AE_SCHEMA));
    Ok(quote! {
        impl #ty {
            #[doc(hidden)]
            pub const __AE_SCHEMA: [u8; #n] = #value;
        }
        unsafe impl #trait_path for #ty {
            const ID: u64 = #id;
            const NAME: &'static str = #name;
            const SCHEMA: &'static [u8] = &<#ty>::__AE_SCHEMA;
        }
        const _: () = {
            fn __ae_component_is_copy<T: ::core::marker::Copy + 'static>() {}
            fn __ae_check() { __ae_component_is_copy::<#ty>(); }
        };
        #section
    })
}

#[derive(Default)]
struct SystemArgs {
    terms: Vec<(Path, u32)>,
    emits: Vec<Path>,
    reads: Vec<Path>,
    after: Vec<Ident>,
    mode: u32,
    cadence: (u32, u32, u32),
}

fn list<T: syn::parse::Parse>(m: &syn::meta::ParseNestedMeta) -> syn::Result<Vec<T>> {
    let content;
    parenthesized!(content in m.input);
    Ok(Punctuated::<T, Token![,]>::parse_terminated(&content)?.into_iter().collect())
}

/// `#[system(...)]` on `fn name(run: &ae_sdk::Run)`. Exports
/// `ae_sys_name(desc: u32)` and records the declaration.
///
/// Arguments, any order: `write(A, ..)`, `read(B, ..)` (query terms, in the
/// order written), `emits(E, ..)`, `reads_events(E, ..)`, `after(system, ..)`,
/// `serial`, and one cadence: `every(n)`, `every(n, phase)`,
/// `staggered(n)` or `once`.
#[proc_macro_attribute]
pub fn system(args: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let mut a = SystemArgs::default();
    let parser = syn::meta::parser(|m| {
        let is = |s: &str| m.path.is_ident(s);
        if is("write") || is("read") {
            let access = if is("write") { abi::ACCESS_WRITE } else { abi::ACCESS_READ };
            for p in list::<Path>(&m)? {
                a.terms.push((p, access));
            }
        } else if is("emits") {
            a.emits.extend(list::<Path>(&m)?);
        } else if is("reads_events") {
            a.reads.extend(list::<Path>(&m)?);
        } else if is("after") {
            a.after.extend(list::<Ident>(&m)?);
        } else if is("serial") {
            a.mode = abi::MODE_SERIAL;
        } else if is("parallel") {
            a.mode = abi::MODE_PARALLEL;
        } else if is("once") {
            a.cadence = (abi::CADENCE_ONCE, 0, 0);
        } else if is("every") || is("staggered") {
            let nums: Vec<u32> = list::<LitInt>(&m)?.iter().map(|l| l.base10_parse()).collect::<syn::Result<_>>()?;
            a.cadence = match (is("every"), nums.as_slice()) {
                (true, [n]) => (abi::CADENCE_EVERY, *n, 0),
                (true, [n, p]) => (abi::CADENCE_EVERY, *n, *p),
                (false, [n]) => (abi::CADENCE_STAGGERED, *n, 0),
                _ => return Err(m.error("expected every(n), every(n, phase) or staggered(n)")),
            };
        } else {
            return Err(m.error("unknown system argument"));
        }
        Ok(())
    });
    parse_macro_input!(args with parser);
    system_impl(func, a).unwrap_or_else(Error::into_compile_error).into()
}

fn system_impl(func: ItemFn, a: SystemArgs) -> syn::Result<Tokens> {
    let name = &func.sig.ident;
    let name_s = name.to_string();
    if a.terms.len() > abi::MAX_TERMS {
        return Err(Error::new_spanned(name, format!("at most {} query terms", abi::MAX_TERMS)));
    }
    if a.reads.len() > abi::MAX_EVENT_READS {
        return Err(Error::new_spanned(name, format!("at most {} event streams read", abi::MAX_EVENT_READS)));
    }
    let mut r = Record::new(abi::KIND_SYSTEM);
    for v in [a.mode, a.cadence.0, a.cadence.1, a.cadence.2, a.terms.len() as u32, a.emits.len() as u32, a.reads.len() as u32, a.after.len() as u32, name_s.len() as u32, 0] {
        r.u32(v);
    }
    r.raw_name(&name_s);
    for (ty, access) in &a.terms {
        r.patch_u64(quote! { <#ty as ::ae_sdk::Component>::ID });
        r.u32(*access);
        r.u32(0);
    }
    for ty in &a.emits {
        r.patch_u64(quote! { <#ty as ::ae_sdk::Event>::ID });
    }
    for ty in &a.reads {
        r.patch_u64(quote! { <#ty as ::ae_sdk::Event>::ID });
    }
    for dep in &a.after {
        r.string(&dep.to_string());
    }
    let (n, value) = r.finish();
    let export = format_ident!("{}{}", abi::SYSTEM_EXPORT_PREFIX, name);
    let section = section_static(&format_ident!("__AE_SCHEMA_SYS_{}", name), n, value);
    Ok(quote! {
        #func
        #[doc(hidden)]
        #[unsafe(no_mangle)]
        pub extern "C" fn #export(desc: u32) {
            let run = unsafe { ::ae_sdk::Run::from_desc(desc) };
            #name(&run);
        }
        #section
    })
}

/// One `schedule!` entry: a module system, or `host(name)` for a system the
/// engine provides (e.g. `host(nav)`).
enum Entry {
    Module(Ident),
    Host(Ident),
}

impl syn::parse::Parse for Entry {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        if name == "host" && input.peek(syn::token::Paren) {
            let content;
            parenthesized!(content in input);
            return Ok(Entry::Host(content.parse()?));
        }
        Ok(Entry::Module(name))
    }
}

/// `schedule!(a, host(nav), b)`: the registration order of the module's
/// systems and of the host systems it uses. Exactly one per module; every
/// module system must be listed.
#[proc_macro]
pub fn schedule(input: TokenStream) -> TokenStream {
    let entries = parse_macro_input!(input with Punctuated::<Entry, Token![,]>::parse_terminated);
    let mut r = Record::new(abi::KIND_ORDER);
    r.u32(entries.len() as u32);
    r.u32(0);
    let mut funcs = Vec::new();
    for e in &entries {
        match e {
            Entry::Module(n) => {
                r.string(&n.to_string());
                funcs.push(n);
            }
            Entry::Host(n) => r.string(&format!("{}{}", abi::HOST_SYSTEM_PREFIX, n)),
        }
    }
    let (n, value) = r.finish();
    let count = funcs.len();
    let section = section_static(&Ident::new("__AE_SCHEMA_ORDER", Span::call_site()), n, value);
    // Naming each function checks at compile time that the systems exist.
    quote! {
        const _: [fn(&::ae_sdk::Run); #count] = [#(#funcs),*];
        #section
    }
    .into()
}
