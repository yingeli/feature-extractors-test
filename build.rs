use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/torch");
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rustc-link-search=native=/usr/local/lib/python3.12/dist-packages/torch/lib");
    println!("cargo:rustc-link-lib=torch_cuda");
    println!("cargo:rustc-link-lib=torch_cpu");
    println!("cargo:rustc-link-lib=torch");
    println!("cargo:rustc-link-lib=c10");
    // println!("cargo:rustc-link-lib=c10_cuda");
    
    cxx_build::bridges([
        "src/torch/log_mel.rs",
    ])
    .file("src/torch/log_mel.cpp")
    .include("src/torch")
    .include("/usr/local/lib/python3.12/dist-packages/torch/include")
    .include("/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include")
    //.cpp(true)
    .std("c++20")
    .cuda(true)
    //.static_flag(true)
    //.static_crt(cfg!(target_os = "windows"))
    .flag_if_supported("/EHsc")
    .flag("-w")
    .compile("feature-extractors-test");
}