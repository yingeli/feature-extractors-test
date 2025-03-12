#include "log_mel.h"
#include "mel_filter.h"

#include <torch/torch.h>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>  // Added for time measurement

LogMelSpectrogram::LogMelSpectrogram(
    const int nMels,
    const int nFFT,
    const int hopLength,
    torch::Device const& device
) : mNFFT(nFFT), 
    mHopLength(hopLength) {
    // Load the mel filter array
    std::cout << "Loading mel filter array..." << std::endl;
    void* melFiltersPtr = const_cast<float*>(MelFiltersArray);
    auto filters = torch::from_blob(melFiltersPtr, {128, 201}, torch::kFloat32);
    std::cout << "Mel filter: " << filters.sizes() << std::endl;

    //filters = filters.t();
    //std::cout << "Mel filter: " << filters.sizes() << std::endl;

    mFilters = filters.to(device);
    std::cout << "Mel filter: " << mFilters.sizes() << std::endl;
    
    mWindow = torch::hann_window(nFFT).to(device);
    std::cout << "Window: " << mWindow.sizes() << std::endl;
}

torch::Tensor LogMelSpectrogram::extract(
    const std::span<const float> samples 
) const {
    auto device = mFilters.device();

    torch::Tensor tensor = torch::from_blob(
        (void*)samples.data(), 
        (long)samples.size(),
        torch::kFloat32).to(device);

    int padding = samples.size() % mHopLength == 0 ? 0 : mHopLength - (samples.size() % mHopLength);
    if (padding > 0) {
        tensor = torch::nn::functional::pad(
            tensor, 
            torch::nn::functional::PadFuncOptions({0, padding}).mode(torch::kConstant).value(0));
    }

    torch::Tensor stft = torch::stft(tensor, 
        mNFFT, 
        mHopLength, 
        mNFFT, 
        mWindow, 
        true, 
        "reflect", 
        false, 
        true, 
        true);

    auto magnitudes = stft.slice(-1, 0, stft.size(stft.dim() - 1) - 1).abs().pow(2);

    torch::Tensor mel_spec = torch::matmul(mFilters, magnitudes);

    torch::Tensor log_spec = torch::clamp_min(mel_spec, 1e-10).log10();
    log_spec = torch::maximum(log_spec, log_spec.max() - 8.0);
    log_spec = (log_spec + 4.0) / 4.0;

    return log_spec;
}

torch::Tensor LogMelSpectrogram::extract_rfft(
    const std::span<const float> samples 
) const {
    auto device = mFilters.device();

    torch::Tensor tensor = torch::from_blob(
        (void*)samples.data(), 
        (long)samples.size(),
        torch::kFloat32).to(device);

    int padding = samples.size() % mHopLength == 0 ? 0 : mHopLength - (samples.size() % mHopLength);
    if (padding > 0) {
        tensor = torch::nn::functional::pad(
            tensor, 
            torch::nn::functional::PadFuncOptions({0, padding}).mode(torch::kConstant).value(0));
    }

    torch::Tensor fft = torch::fft::rfft(tensor);

    auto magnitudes = fft.slice(-1, 0, fft.size(fft.dim() - 1) - 1).abs().pow(2);

    torch::Tensor mel_spec = torch::matmul(mFilters, magnitudes);

    torch::Tensor log_spec = torch::clamp_min(mel_spec, 1e-10).log10();
    log_spec = torch::maximum(log_spec, log_spec.max() - 8.0);
    log_spec = (log_spec + 4.0) / 4.0;

    return log_spec;
}