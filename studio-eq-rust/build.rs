use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_rc_exe() -> Option<PathBuf> {
    if let Ok(out) = Command::new("where").arg("rc.exe").output() {
        if out.status.success() {
            if let Some(line) = String::from_utf8_lossy(&out.stdout).lines().next() {
                let p = PathBuf::from(line.trim());
                if p.exists() {
                    return Some(p);
                }
            }
        }
    }

    let kits_bin = Path::new(r"C:\Program Files (x86)\Windows Kits\10\bin");
    if let Ok(entries) = fs::read_dir(kits_bin) {
        let mut candidates: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path().join("x64").join("rc.exe"))
            .filter(|p| p.exists())
            .collect();
        candidates.sort();
        if let Some(best) = candidates.pop() {
            return Some(best);
        }
    }
    None
}

/// Pure-Rust Win32 `.res` generator fallback from a multi-entry `.ico` file
/// (embeds RT_ICON = 3 and RT_GROUP_ICON = 14 with resource ID 1).
fn build_res_from_ico(ico_bytes: &[u8]) -> Option<Vec<u8>> {
    if ico_bytes.len() < 6 {
        return None;
    }
    let reserved = u16::from_le_bytes([ico_bytes[0], ico_bytes[1]]);
    let ico_type = u16::from_le_bytes([ico_bytes[2], ico_bytes[3]]);
    let count = u16::from_le_bytes([ico_bytes[4], ico_bytes[5]]) as usize;
    if reserved != 0 || ico_type != 1 || count == 0 || ico_bytes.len() < 6 + count * 16 {
        return None;
    }

    let mut res = Vec::new();

    let push_aligned4 = |buf: &mut Vec<u8>| {
        while buf.len() % 4 != 0 {
            buf.push(0);
        }
    };

    let write_res_header = |buf: &mut Vec<u8>, data_size: u32, res_type: u16, res_id: u16| {
        let header_size: u32 = 32;
        buf.extend_from_slice(&data_size.to_le_bytes());
        buf.extend_from_slice(&header_size.to_le_bytes());
        // Ordinal type: 0xFFFF, type_id
        buf.extend_from_slice(&0xFFFFu16.to_le_bytes());
        buf.extend_from_slice(&res_type.to_le_bytes());
        // Ordinal name: 0xFFFF, name_id
        buf.extend_from_slice(&0xFFFFu16.to_le_bytes());
        buf.extend_from_slice(&res_id.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // DataVersion
        buf.extend_from_slice(&0x1030u16.to_le_bytes()); // MemoryFlags
        buf.extend_from_slice(&0x0409u16.to_le_bytes()); // LanguageId (en-US neutral)
        buf.extend_from_slice(&0u32.to_le_bytes()); // Version
        buf.extend_from_slice(&0u32.to_le_bytes()); // Characteristics
    };

    // 32-byte mandatory empty prelude resource in Win32 .res format
    res.extend_from_slice(&0u32.to_le_bytes());
    res.extend_from_slice(&32u32.to_le_bytes());
    res.extend_from_slice(&0xFFFFu16.to_le_bytes());
    res.extend_from_slice(&0u16.to_le_bytes());
    res.extend_from_slice(&0xFFFFu16.to_le_bytes());
    res.extend_from_slice(&0u16.to_le_bytes());
    res.extend_from_slice(&[0u8; 16]);

    let mut grp_data = Vec::with_capacity(6 + count * 14);
    grp_data.extend_from_slice(&0u16.to_le_bytes());
    grp_data.extend_from_slice(&1u16.to_le_bytes());
    grp_data.extend_from_slice(&(count as u16).to_le_bytes());

    for i in 0..count {
        let base = 6 + i * 16;
        let entry = &ico_bytes[base..base + 16];
        let bytes_in_res = u32::from_le_bytes([entry[8], entry[9], entry[10], entry[11]]) as usize;
        let image_offset =
            u32::from_le_bytes([entry[12], entry[13], entry[14], entry[15]]) as usize;
        if image_offset + bytes_in_res > ico_bytes.len() {
            return None;
        }
        let icon_id = (i + 1) as u16;
        // GRPICONDIRENTRY: first 12 bytes identical to ICONDIRENTRY, followed by u16 nID
        grp_data.extend_from_slice(&entry[0..12]);
        grp_data.extend_from_slice(&icon_id.to_le_bytes());

        // Write RT_ICON (type = 3)
        write_res_header(&mut res, bytes_in_res as u32, 3, icon_id);
        res.extend_from_slice(&ico_bytes[image_offset..image_offset + bytes_in_res]);
        push_aligned4(&mut res);
    }

    // Write RT_GROUP_ICON (type = 14) with ID = 1 (primary executable icon in Windows Explorer)
    write_res_header(&mut res, grp_data.len() as u32, 14, 1);
    res.extend_from_slice(&grp_data);
    push_aligned4(&mut res);

    Some(res)
}

fn main() {
    println!("cargo:rerun-if-changed=../app_icon.ico");
    println!("cargo:rerun-if-changed=build.rs");

    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() != "windows" {
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let ico_path = manifest_dir.parent().unwrap().join("app_icon.ico");
    if !ico_path.exists() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let res_path = out_dir.join("studio_eq_icon.res");

    let mut compiled = false;
    if let Some(rc_exe) = find_rc_exe() {
        let rc_file = out_dir.join("studio_eq_icon.rc");
        let escaped_ico = ico_path.to_string_lossy().replace('\\', "/");
        let rc_content = format!("1 ICON \"{escaped_ico}\"\r\n");
        if fs::write(&rc_file, rc_content).is_ok() {
            if let Ok(status) = Command::new(rc_exe)
                .args(["/nologo", "/fo"])
                .arg(&res_path)
                .arg(&rc_file)
                .status()
            {
                compiled = status.success() && res_path.exists();
            }
        }
    }

    if !compiled {
        if let Ok(ico_bytes) = fs::read(&ico_path) {
            if let Some(res_bytes) = build_res_from_ico(&ico_bytes) {
                compiled = fs::write(&res_path, res_bytes).is_ok();
            }
        }
    }

    if compiled {
        println!("cargo:rustc-link-arg-bins={}", res_path.display());
    }
}
