use anyhow::{anyhow, Result};
use libloading::Library;
use serde::Serialize;
use std::{
    ffi::{c_void, CStr},
    fs,
    process::{Command, Output},
};

type CuResult = i32;
type CuDevice = i32;
type CuInit = unsafe extern "C" fn(u32) -> CuResult;
type CuDriverGetVersion = unsafe extern "C" fn(*mut i32) -> CuResult;
type CuDeviceGetCount = unsafe extern "C" fn(*mut i32) -> CuResult;
type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, i32) -> CuResult;
type CuDeviceGetName = unsafe extern "C" fn(*mut i8, i32, CuDevice) -> CuResult;
type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult;
type CuDeviceGetUuid = unsafe extern "C" fn(*mut CuUuid, CuDevice) -> CuResult;
type CuDeviceGetAttribute = unsafe extern "C" fn(*mut i32, i32, CuDevice) -> CuResult;

const CUDA_SUCCESS: CuResult = 0;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

#[repr(C)]
#[derive(Clone, Copy)]
struct CuUuid {
    bytes: [u8; 16],
}

#[derive(Clone, Debug, Serialize)]
struct Check {
    available: bool,
    detail: String,
}

#[derive(Debug, Serialize)]
struct Device {
    ordinal: i32,
    name: String,
    uuid: String,
    memory_bytes: Option<u64>,
    compute_capability: Option<String>,
}

#[derive(Debug, Serialize)]
struct CudaReport {
    library: Check,
    driver: Check,
    persistence_mode: Check,
    checkpoint_api: Check,
    device_count: Option<i32>,
    devices: Vec<Device>,
}

#[derive(Debug, Serialize)]
struct ProcessPermissions {
    effective_uid: Option<u32>,
    ptrace: Check,
    namespaces: Check,
    criu_capabilities: Check,
}

#[derive(Debug, Serialize)]
struct Report {
    platform: String,
    architecture: String,
    linux_checkpoint_features: Check,
    criu: Check,
    process_permissions: ProcessPermissions,
    cuda: CudaReport,
}

pub fn run(json: bool) -> Result<()> {
    let report = collect();

    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human(&report);
    }

    Ok(())
}

fn collect() -> Report {
    let criu = check_criu();
    Report {
        platform: std::env::consts::OS.to_owned(),
        architecture: std::env::consts::ARCH.to_owned(),
        linux_checkpoint_features: if cfg!(target_os = "linux") {
            Check {
                available: true,
                detail: "Linux checkpoint features can be discovered".to_owned(),
            }
        } else {
            Check {
                available: false,
                detail: "CUDA checkpoint and CRIU integration require Linux".to_owned(),
            }
        },
        criu: criu.clone(),
        process_permissions: check_process_permissions(&criu),
        cuda: check_cuda(),
    }
}

fn check_process_permissions(criu: &Check) -> ProcessPermissions {
    ProcessPermissions {
        effective_uid: read_effective_uid(),
        ptrace: check_ptrace_scope(),
        namespaces: check_namespaces(),
        criu_capabilities: Check {
            available: criu.available,
            detail: format!("CRIU capability check: {}", criu.detail),
        },
    }
}

fn read_effective_uid() -> Option<u32> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let values = line
            .strip_prefix("Uid:")?
            .split_whitespace()
            .collect::<Vec<_>>();
        values.get(1)?.parse().ok()
    })
}

fn check_ptrace_scope() -> Check {
    match fs::read_to_string("/proc/sys/kernel/yama/ptrace_scope") {
        Ok(scope) => Check {
            available: true,
            detail: format!("ptrace_scope={} (readable)", scope.trim()),
        },
        Err(error) => unavailable(format!("could not read ptrace_scope: {error}")),
    }
}

fn check_namespaces() -> Check {
    let names = ["pid", "mnt", "user"];
    let missing = names
        .iter()
        .filter(|name| fs::read_link(format!("/proc/self/ns/{name}")).is_err())
        .copied()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Check {
            available: true,
            detail: "PID, mount, and user namespaces are readable".to_owned(),
        }
    } else {
        unavailable(format!("unreadable namespaces: {}", missing.join(", ")))
    }
}

fn check_criu() -> Check {
    let version = Command::new("criu").arg("--version").output();
    let Ok(version) = version else {
        return unavailable("criu executable not found on PATH");
    };

    if !version.status.success() {
        return unavailable(format!("criu --version failed with {}", version.status));
    }

    let version_text = command_output(&version);
    match Command::new("criu").arg("check").output() {
        Ok(output) if output.status.success() => Check {
            available: true,
            detail: format!("{version_text}; kernel capability check passed"),
        },
        Ok(output) => unavailable(format!(
            "{version_text}; kernel capability check failed: {}",
            command_output(&output)
        )),
        Err(error) => unavailable(format!("{version_text}; could not run criu check: {error}")),
    }
}

fn check_cuda() -> CudaReport {
    let library = unsafe { Library::new("libcuda.so.1").or_else(|_| Library::new("libcuda.so")) };
    let Ok(library) = library else {
        return CudaReport {
            library: unavailable("libcuda.so.1/libcuda.so not found"),
            driver: unavailable("CUDA driver could not be loaded"),
            persistence_mode: check_persistence_mode(),
            checkpoint_api: unavailable("CUDA checkpoint API cannot be inspected"),
            device_count: None,
            devices: Vec::new(),
        };
    };

    let checkpoint_symbols: [(&str, &[u8]); 6] = [
        ("cuCheckpointProcessLock", b"cuCheckpointProcessLock\0"),
        (
            "cuCheckpointProcessCheckpoint",
            b"cuCheckpointProcessCheckpoint\0",
        ),
        (
            "cuCheckpointProcessRestore",
            b"cuCheckpointProcessRestore\0",
        ),
        ("cuCheckpointProcessUnlock", b"cuCheckpointProcessUnlock\0"),
        (
            "cuCheckpointProcessGetState",
            b"cuCheckpointProcessGetState\0",
        ),
        (
            "cuCheckpointProcessGetRestoreThreadId",
            b"cuCheckpointProcessGetRestoreThreadId\0",
        ),
    ];
    let missing: Vec<_> = checkpoint_symbols
        .iter()
        .filter(|(_, symbol)| unsafe { library.get::<*const c_void>(symbol).is_err() })
        .map(|(name, _)| *name)
        .collect();

    let checkpoint_api = if missing.is_empty() {
        Check {
            available: true,
            detail: "all required CUDA checkpoint symbols are exported".to_owned(),
        }
    } else {
        unavailable(format!("missing symbols: {}", missing.join(", ")))
    };

    let library_check = Check {
        available: true,
        detail: "loaded libcuda.so.1 or libcuda.so".to_owned(),
    };

    let result = unsafe { query_cuda(&library) };
    match result {
        Ok((driver, devices)) => CudaReport {
            library: library_check,
            driver: Check {
                available: true,
                detail: driver,
            },
            persistence_mode: check_persistence_mode(),
            checkpoint_api,
            device_count: Some(devices.len() as i32),
            devices,
        },
        Err(error) => CudaReport {
            library: library_check,
            driver: unavailable(error.to_string()),
            persistence_mode: check_persistence_mode(),
            checkpoint_api,
            device_count: None,
            devices: Vec::new(),
        },
    }
}

fn check_persistence_mode() -> Check {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=persistence_mode",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let Ok(output) = output else {
        return unavailable("nvidia-smi unavailable; persistence mode could not be checked");
    };
    if !output.status.success() {
        return unavailable("nvidia-smi could not query persistence mode");
    }
    let value = command_output(&output);
    if value.eq_ignore_ascii_case("enabled") {
        Check {
            available: true,
            detail: "persistence mode is enabled".to_owned(),
        }
    } else {
        Check {
            available: false,
            detail: format!(
                "persistence mode is {value}; restore may require CUDA reinitialization"
            ),
        }
    }
}

unsafe fn query_cuda(library: &Library) -> Result<(String, Vec<Device>)> {
    let init: libloading::Symbol<CuInit> = library.get(b"cuInit\0")?;
    let driver_get_version: libloading::Symbol<CuDriverGetVersion> =
        library.get(b"cuDriverGetVersion\0")?;
    let device_get_count: libloading::Symbol<CuDeviceGetCount> =
        library.get(b"cuDeviceGetCount\0")?;
    let device_get: libloading::Symbol<CuDeviceGet> = library.get(b"cuDeviceGet\0")?;
    let device_get_name: libloading::Symbol<CuDeviceGetName> = library.get(b"cuDeviceGetName\0")?;
    let device_total_mem: libloading::Symbol<CuDeviceTotalMem> =
        library.get(b"cuDeviceTotalMem_v2\0")?;
    let device_get_uuid: libloading::Symbol<CuDeviceGetUuid> = library.get(b"cuDeviceGetUuid\0")?;
    let device_get_attribute: libloading::Symbol<CuDeviceGetAttribute> =
        library.get(b"cuDeviceGetAttribute\0")?;

    cuda_ok(init(0), "cuInit")?;

    let mut raw_driver_version = 0;
    cuda_ok(
        driver_get_version(&mut raw_driver_version),
        "cuDriverGetVersion",
    )?;
    let driver = format!(
        "CUDA driver API {}.{} (raw {})",
        raw_driver_version / 1000,
        (raw_driver_version % 1000) / 10,
        raw_driver_version
    );

    let mut count = 0;
    cuda_ok(device_get_count(&mut count), "cuDeviceGetCount")?;
    let mut devices = Vec::with_capacity(count.max(0) as usize);

    for ordinal in 0..count {
        let mut device = 0;
        cuda_ok(device_get(&mut device, ordinal), "cuDeviceGet")?;

        let mut name = [0i8; 256];
        cuda_ok(
            device_get_name(name.as_mut_ptr(), name.len() as i32, device),
            "cuDeviceGetName",
        )?;
        let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();

        let mut uuid = CuUuid { bytes: [0; 16] };
        let uuid = if cuda_ok(device_get_uuid(&mut uuid, device), "cuDeviceGetUuid").is_ok() {
            uuid.bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        } else {
            "unavailable".to_owned()
        };

        let mut memory = 0usize;
        let memory_bytes =
            if cuda_ok(device_total_mem(&mut memory, device), "cuDeviceTotalMem_v2").is_ok() {
                Some(memory as u64)
            } else {
                None
            };

        let major = device_attribute(
            &device_get_attribute,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        );
        let minor = device_attribute(
            &device_get_attribute,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        );
        let compute_capability = major
            .zip(minor)
            .map(|(major, minor)| format!("{major}.{minor}"));

        devices.push(Device {
            ordinal,
            name,
            uuid,
            memory_bytes,
            compute_capability,
        });
    }

    Ok((driver, devices))
}

unsafe fn device_attribute(
    function: &libloading::Symbol<'_, CuDeviceGetAttribute>,
    attribute: i32,
    device: CuDevice,
) -> Option<i32> {
    let mut value = 0;
    (function)(&mut value, attribute, device)
        .eq(&CUDA_SUCCESS)
        .then_some(value)
}

fn cuda_ok(result: CuResult, operation: &str) -> Result<()> {
    if result == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(anyhow!("{operation} returned CUDA error code {result}"))
    }
}

fn unavailable(detail: impl Into<String>) -> Check {
    Check {
        available: false,
        detail: detail.into(),
    }
}

fn command_output(output: &Output) -> String {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    format!("{stdout}{stderr}").trim().replace('\n', " ")
}

fn print_human(report: &Report) {
    println!("Edo Tensei capability report");
    println!("  platform: {}", report.platform);
    println!("  architecture: {}", report.architecture);
    print_check(
        "Linux checkpoint features",
        &report.linux_checkpoint_features,
    );
    print_check("CRIU", &report.criu);
    println!(
        "  effective UID: {}",
        report
            .process_permissions
            .effective_uid
            .map_or_else(|| "unknown".to_owned(), |uid| uid.to_string())
    );
    print_check("ptrace", &report.process_permissions.ptrace);
    print_check("namespaces", &report.process_permissions.namespaces);
    print_check(
        "CRIU capabilities",
        &report.process_permissions.criu_capabilities,
    );
    print_check("CUDA library", &report.cuda.library);
    print_check("CUDA driver", &report.cuda.driver);
    print_check("CUDA persistence mode", &report.cuda.persistence_mode);
    print_check("CUDA checkpoint API", &report.cuda.checkpoint_api);
    println!(
        "  CUDA devices: {}",
        report
            .cuda
            .device_count
            .map_or_else(|| "unknown".to_owned(), |count| count.to_string())
    );

    for device in &report.cuda.devices {
        println!("    GPU {}: {}", device.ordinal, device.name);
        println!("      UUID: {}", device.uuid);
        println!(
            "      memory: {}",
            device
                .memory_bytes
                .map_or_else(|| "unknown".to_owned(), format_bytes)
        );
        println!(
            "      compute capability: {}",
            device.compute_capability.as_deref().unwrap_or("unknown")
        );
    }
}

fn print_check(label: &str, check: &Check) {
    let status = if check.available { "OK" } else { "FAIL" };
    println!("  {label}: {status} ({})", check.detail);
}

fn format_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    format!("{:.1} GiB", bytes as f64 / GIB as f64)
}
