#![recursion_limit = "1024"]

#[macro_use]
extern crate error_chain;

pub mod errors {
    error_chain! {
        foreign_links {
            Io(::std::io::Error);
            Utf8Error(::std::string::FromUtf8Error);
        }
        // Define additional `ErrorKind` variants.  Define custom responses with the
        // `description` and `display` calls.
        errors {
            UnsupportedModelType(t: String) {
                description("Unsupported model type")
                display("Unsupported model type: '{}'", t)
            }
            UnsupportedObjFunctionType(t: String) {
                description("Unsupported object function type")
                display("Unsupported object function type: '{}'", t)
            }
        }
    }
}

mod functions;
pub mod fvec;
mod gbm;
pub mod model_reader;
pub mod predictor;
