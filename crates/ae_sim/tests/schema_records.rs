//! Schema records written by the SDK derive (compiled natively here) and
//! read back by the host parser, plus the parser's rejections.

use ae_sdk::abi;
use ae_sdk::{Component, Entity, Event};
use ae_sim::schedule::{Cadence, Mode};
use ae_sim::schema::Schema;

#[repr(C)]
#[derive(Clone, Copy, Component)]
pub struct Body {
    pub pos: [f32; 2],
    #[field(unhashed)]
    pub scratch: u8,
    pub owner: Entity,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(id = 77)]
pub struct Tagged {
    #[field(id = 5)]
    pub value: i16,
}

#[repr(C)]
#[derive(Clone, Copy, Component)]
#[component(event)]
pub struct Hit {
    pub target: Entity,
    pub damage: f32,
}

fn string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
    out.resize(abi::pad4(out.len()), 0);
}

fn record(kind: u32, body: &[u8]) -> Vec<u8> {
    let mut r = Vec::new();
    r.extend_from_slice(&kind.to_le_bytes());
    r.extend_from_slice(&((8 + body.len()) as u32).to_le_bytes());
    r.extend_from_slice(body);
    r
}

fn system(name: &str, terms: &[(u64, u32)], emits: &[u64], reads: &[u64], cadence: (u32, u32, u32)) -> Vec<u8> {
    let mut b = Vec::new();
    for v in [
        abi::MODE_SERIAL,
        cadence.0,
        cadence.1,
        cadence.2,
        terms.len() as u32,
        emits.len() as u32,
        reads.len() as u32,
        0,
        name.len() as u32,
        0,
    ] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(name.as_bytes());
    b.resize(abi::pad4(b.len()), 0);
    for &(id, access) in terms {
        b.extend_from_slice(&id.to_le_bytes());
        b.extend_from_slice(&access.to_le_bytes());
        b.extend_from_slice(&0u32.to_le_bytes());
    }
    for id in emits.iter().chain(reads) {
        b.extend_from_slice(&id.to_le_bytes());
    }
    record(abi::KIND_SYSTEM, &b)
}

fn order(names: &[&str]) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&(names.len() as u32).to_le_bytes());
    b.extend_from_slice(&0u32.to_le_bytes());
    for n in names {
        string(&mut b, n);
    }
    record(abi::KIND_ORDER, &b)
}

fn section(parts: &[&[u8]]) -> Vec<u8> {
    parts.concat()
}

#[test]
fn derive_records_parse_back() {
    let sys = system(
        "apply",
        &[(Body::ID, abi::ACCESS_WRITE), (Tagged::ID, abi::ACCESS_READ)],
        &[],
        &[Hit::ID],
        (abi::CADENCE_EVERY, 3, 1),
    );
    // Record order must not matter.
    let a = Schema::parse(&section(&[
        Body::SCHEMA,
        &sys,
        Tagged::SCHEMA,
        <Hit as Event>::SCHEMA,
        &order(&["apply"]),
    ]))
    .unwrap();
    let b = Schema::parse(&section(&[
        &order(&["apply"]),
        <Hit as Event>::SCHEMA,
        Tagged::SCHEMA,
        &sys,
        Body::SCHEMA,
    ]))
    .unwrap();
    assert_eq!(a.components, b.components);
    assert_eq!(a.systems, b.systems);

    let body = a.component(Body::ID).unwrap();
    assert_eq!(Body::ID, abi::name_id("Body"));
    assert_eq!(
        (body.name.as_str(), body.size, body.align, body.event),
        ("Body", size_of::<Body>() as u32, align_of::<Body>() as u32, false)
    );
    let f: Vec<_> = body
        .fields
        .iter()
        .map(|f| (f.name.as_str(), f.offset, f.size, f.kind, f.hashed()))
        .collect();
    assert_eq!(
        f,
        [
            ("pos", 0, 8, abi::FIELD_F32 | abi::FIELD_ARRAY, true),
            ("scratch", 8, 1, abi::FIELD_U8, false),
            (
                "owner",
                std::mem::offset_of!(Body, owner) as u32,
                8,
                abi::FIELD_ENTITY,
                true
            ),
        ]
    );
    let tagged = a.component(77).unwrap();
    assert_eq!((tagged.fields[0].id, tagged.fields[0].kind), (5, abi::FIELD_I16));
    assert!(a.component(Hit::ID).unwrap().event);

    let s = &a.systems[0];
    assert_eq!(
        (s.name.as_str(), s.mode, s.cadence),
        ("apply", Mode::Serial, Cadence::Every { n: 3, phase: 1 })
    );
    assert_eq!(s.reads_events, [Hit::ID]);
    assert_eq!(s.export(), "ae_sys_apply");
}

#[test]
fn bad_schemas_are_rejected() {
    let ok_sys = system("s", &[(Body::ID, abi::ACCESS_READ)], &[], &[], (0, 0, 0));
    let err = |parts: &[&[u8]]| Schema::parse(&section(parts)).unwrap_err().to_string();
    assert!(err(&[Body::SCHEMA, &ok_sys]).contains("no schedule!"));
    assert!(err(&[Body::SCHEMA, &ok_sys, &order(&["s", "t"])]).contains("`t`"));
    assert!(err(&[Body::SCHEMA, &ok_sys, &order(&[])]).contains("missing from schedule!"));
    assert!(err(&[&ok_sys, &order(&["s"])]).contains("unknown component"));
    let uses_event = system("s", &[(Hit::ID, abi::ACCESS_READ)], &[], &[], (0, 0, 0));
    assert!(err(&[<Hit as Event>::SCHEMA, &uses_event, &order(&["s"])]).contains("queries event type"));
    let emits_component = system("s", &[], &[Body::ID], &[], (0, 0, 0));
    assert!(err(&[Body::SCHEMA, &emits_component, &order(&["s"])]).contains("as an event stream"));
    // Same id, different layout.
    let mut clash = Tagged::SCHEMA.to_vec();
    clash[abi::component::SIZE] ^= 4;
    assert!(Schema::parse(&section(&[Tagged::SCHEMA, &clash, &order(&[])])).is_err());
    // Truncated record.
    let body = Body::SCHEMA;
    assert!(Schema::parse(&body[..body.len() - 4]).is_err());
    // Identical duplicates (the same type seen twice) are fine.
    assert!(Schema::parse(&section(&[Body::SCHEMA, Body::SCHEMA, &order(&[])])).is_ok());
}
