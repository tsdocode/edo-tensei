//! Dynamically loaded CUDA process checkpoint API.
#![allow(dead_code)]

use anyhow::{anyhow, bail, Context, Result};
use libloading::Library;
use std::{
    ffi::CStr,
    os::raw::c_char,
    thread,
    time::{Duration, Instant},
};

type CuResult = i32;
type CuInit = unsafe extern "C" fn(u32) -> CuResult;
type CuErrorName = unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult;
type CuRestoreThreadId = unsafe extern "C" fn(i32, *mut i32) -> CuResult;
type CuGetState = unsafe extern "C" fn(i32, *mut i32) -> CuResult;
type CuLock = unsafe extern "C" fn(i32, *mut LockArgs) -> CuResult;
type CuCheckpoint = unsafe extern "C" fn(i32, *mut OpArgs) -> CuResult;
type CuRestore = unsafe extern "C" fn(i32, *mut OpArgs) -> CuResult;
type CuUnlock = unsafe extern "C" fn(i32, *mut OpArgs) -> CuResult;

const CUDA_SUCCESS: CuResult = 0;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessState {
    Running = 0,
    Locked = 1,
    Checkpointed = 2,
    Failed = 3,
    Unknown(i32),
}

impl ProcessState {
    fn from_raw(value: i32) -> Self {
        match value {
            0 => Self::Running,
            1 => Self::Locked,
            2 => Self::Checkpointed,
            3 => Self::Failed,
            value => Self::Unknown(value),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LockArgs {
    timeout_ms: u32,
    reserved0: u32,
    reserved1: [u64; 7],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct OpArgs {
    reserved: [u64; 8],
}

impl LockArgs {
    fn with_timeout(timeout_ms: u32) -> Self {
        Self {
            timeout_ms,
            reserved0: 0,
            reserved1: [0; 7],
        }
    }
}

#[derive(Debug)]
pub struct CudaCheckpoint {
    _library: Library,
    error_name: Option<CuErrorName>,
    init: CuInit,
    restore_thread_id: CuRestoreThreadId,
    get_state: CuGetState,
    lock: CuLock,
    checkpoint: CuCheckpoint,
    restore: CuRestore,
    unlock: CuUnlock,
}

// The loaded library is kept alive by the owning CudaCheckpoint, while all
// symbols are immutable function pointers. CUDA's process-checkpoint API is
// process-ID based, so independent owner PIDs can be driven concurrently.
// The group restore path uses this only after the object is fully initialized.
unsafe impl Send for CudaCheckpoint {}
unsafe impl Sync for CudaCheckpoint {}

impl CudaCheckpoint {
    pub fn load() -> Result<Self> {
        let library = unsafe {
            Library::new("libcuda.so.1")
                .or_else(|_| Library::new("libcuda.so"))
                .context("could not load libcuda.so.1 or libcuda.so")?
        };
        unsafe {
            Ok(Self {
                error_name: library
                    .get::<CuErrorName>(b"cuGetErrorName\0")
                    .ok()
                    .map(|s| *s),
                init: *library
                    .get(b"cuInit\0")
                    .context("missing CUDA symbol cuInit")?,
                restore_thread_id: *library
                    .get(b"cuCheckpointProcessGetRestoreThreadId\0")
                    .context(
                        "missing CUDA checkpoint symbol cuCheckpointProcessGetRestoreThreadId",
                    )?,
                get_state: *library
                    .get(b"cuCheckpointProcessGetState\0")
                    .context("missing CUDA checkpoint symbol cuCheckpointProcessGetState")?,
                lock: *library
                    .get(b"cuCheckpointProcessLock\0")
                    .context("missing CUDA checkpoint symbol cuCheckpointProcessLock")?,
                checkpoint: *library
                    .get(b"cuCheckpointProcessCheckpoint\0")
                    .context("missing CUDA checkpoint symbol cuCheckpointProcessCheckpoint")?,
                restore: *library
                    .get(b"cuCheckpointProcessRestore\0")
                    .context("missing CUDA checkpoint symbol cuCheckpointProcessRestore")?,
                unlock: *library
                    .get(b"cuCheckpointProcessUnlock\0")
                    .context("missing CUDA checkpoint symbol cuCheckpointProcessUnlock")?,
                _library: library,
            })
        }
    }

    pub fn initialize(&self) -> Result<()> {
        self.call(unsafe { (self.init)(0) }, "cuInit")
    }

    pub fn state(&self, pid: i32) -> Result<ProcessState> {
        let mut state = 0;
        self.call(
            unsafe { (self.get_state)(pid, &mut state) },
            "cuCheckpointProcessGetState",
        )?;
        Ok(ProcessState::from_raw(state))
    }

    pub fn wait_for_state(
        &self,
        pid: i32,
        expected: ProcessState,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            let current = self.state(pid)?;
            if current == expected {
                return Ok(());
            }
            if current == ProcessState::Failed {
                bail!("CUDA process {pid} entered FAILED state");
            }
            if Instant::now() >= deadline {
                bail!(
                    "CUDA process {pid} did not reach {expected:?}; current state is {current:?}"
                );
            }
            thread::sleep(poll_interval);
        }
    }

    pub fn restore_thread_id(&self, pid: i32) -> Result<i32> {
        let mut tid = 0;
        self.call(
            unsafe { (self.restore_thread_id)(pid, &mut tid) },
            "cuCheckpointProcessGetRestoreThreadId",
        )?;
        Ok(tid)
    }

    pub fn lock(&self, pid: i32, timeout_ms: u32) -> Result<()> {
        let mut args = LockArgs::with_timeout(timeout_ms);
        self.call(
            unsafe { (self.lock)(pid, &mut args) },
            "cuCheckpointProcessLock",
        )
    }

    pub fn checkpoint(&self, pid: i32) -> Result<()> {
        let mut args = OpArgs { reserved: [0; 8] };
        self.call(
            unsafe { (self.checkpoint)(pid, &mut args) },
            "cuCheckpointProcessCheckpoint",
        )
    }

    pub fn restore(&self, pid: i32) -> Result<()> {
        let mut args = OpArgs { reserved: [0; 8] };
        self.call(
            unsafe { (self.restore)(pid, &mut args) },
            "cuCheckpointProcessRestore",
        )
    }

    pub fn unlock(&self, pid: i32) -> Result<()> {
        let mut args = OpArgs { reserved: [0; 8] };
        self.call(
            unsafe { (self.unlock)(pid, &mut args) },
            "cuCheckpointProcessUnlock",
        )
    }

    fn call(&self, result: CuResult, operation: &str) -> Result<()> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }
        let mut name = std::ptr::null();
        let label = match self.error_name {
            Some(get_name)
                if unsafe { get_name(result, &mut name) } == CUDA_SUCCESS && !name.is_null() =>
            {
                unsafe { CStr::from_ptr(name) }
                    .to_string_lossy()
                    .into_owned()
            }
            _ => "unknown".to_owned(),
        };
        Err(anyhow!(
            "{operation} failed with CUDA error {result} ({label})"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_argument_abis_match_cuda_12_8() {
        assert_eq!(std::mem::size_of::<LockArgs>(), 64);
        assert_eq!(std::mem::size_of::<OpArgs>(), 64);
    }

    #[test]
    fn process_states_preserve_unknown_values() {
        assert_eq!(ProcessState::from_raw(0), ProcessState::Running);
        assert_eq!(ProcessState::from_raw(2), ProcessState::Checkpointed);
        assert_eq!(ProcessState::from_raw(99), ProcessState::Unknown(99));
    }
}
