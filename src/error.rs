use thiserror::Error;

// These variants are defined for the upcoming capability and snapshot
// implementations. Keep the scaffold warning-free until they are consumed.
#[allow(dead_code)]
#[derive(Debug, Error)]
pub enum EdoError {
    #[error("this operation requires Linux")]
    UnsupportedPlatform,

    #[error("required executable is not available: {0}")]
    MissingExecutable(String),

    #[error("required library or symbol is not available: {0}")]
    MissingLibrary(String),

    #[error("snapshot is incompatible: {0}")]
    SnapshotIncompatible(String),

    #[error("invalid state transition: {0}")]
    InvalidState(String),
}
