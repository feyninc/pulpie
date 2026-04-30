use thiserror::Error;

pub type Result<T> = std::result::Result<T, HummingbirdError>;

#[derive(Error, Debug)]
pub enum HummingbirdError {
    #[error("No content found")]
    NoContent,

    #[error("Extraction error: {0}")]
    Extraction(String),
}
