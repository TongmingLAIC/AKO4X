#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// The host and CPU oracle are frozen by the example campaign's editable scope.
// A build only checks Objective-C++; Metal compilation happens at runtime.
int main(int argc, char** argv) {
    @autoreleasepool {
        MTLCaptureManager* capture_manager = nil;
        try {
            uint32_t rows = 4, columns = 128, seed = 17;
            bool bench = false;
            std::string output, capture;
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--help") {
                    std::cout << "rmsnorm_bench --check|--bench [--rows N] [--columns N] [--seed N] [--output FILE] [--capture NEW.gputrace]\n";
                    return 0;
                }
                if (arg == "--bench") { bench = true; continue; }
                if (arg == "--check") continue;
                if (++i >= argc) throw std::runtime_error("missing argument value");
                if (arg == "--output") { output = argv[i]; continue; }
                if (arg == "--capture") { capture = argv[i]; continue; }
                const auto value = std::stoul(argv[i]);
                if (value > UINT32_MAX) throw std::runtime_error("argument too large");
                if (arg == "--rows") rows = uint32_t(value);
                else if (arg == "--columns") columns = uint32_t(value);
                else if (arg == "--seed") seed = uint32_t(value);
                else throw std::runtime_error("unknown argument: " + arg);
            }
            const size_t count = size_t(rows) * columns;
            if (!rows || !columns || count > 16 * 1024 * 1024)
                throw std::runtime_error("shape must contain 1..16777216 elements");
            if (!capture.empty() && (std::filesystem::exists(capture) ||
                                    std::filesystem::path(capture).extension() != ".gputrace"))
                throw std::runtime_error("capture must name a new .gputrace path");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (!device) throw std::runtime_error("no Metal device");
            NSError* error = nil;
            NSString* source = [NSString stringWithContentsOfFile:@AKO_KERNEL_SOURCE
                                                       encoding:NSUTF8StringEncoding error:&error];
            if (!source) throw std::runtime_error([[error description] UTF8String]);
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
            if (!library) throw std::runtime_error([[error description] UTF8String]);
            id<MTLFunction> function = [library newFunctionWithName:@"rmsnorm"];
            if (!function) throw std::runtime_error("missing rmsnorm entry point");
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) throw std::runtime_error([[error description] UTF8String]);
            id<MTLCommandQueue> queue = [device newCommandQueue];
            id<MTLBuffer> x = [device newBufferWithLength:count * sizeof(float) options:MTLResourceStorageModeShared];
            id<MTLBuffer> w = [device newBufferWithLength:columns * sizeof(float) options:MTLResourceStorageModeShared];
            constexpr size_t guard = 16;
            id<MTLBuffer> y = [device newBufferWithLength:(count + 2 * guard) * sizeof(float) options:MTLResourceStorageModeShared];
            if (!queue || !x || !w || !y) throw std::runtime_error("Metal allocation failed");
            if (!capture.empty()) {
                capture_manager = [MTLCaptureManager sharedCaptureManager];
                if (![capture_manager supportsDestination:MTLCaptureDestinationGPUTraceDocument])
                    throw std::runtime_error("GPU trace capture unavailable; check tooling and MTL_CAPTURE_ENABLED");
                MTLCaptureDescriptor* descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = queue;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:@(capture.c_str())];
                if (![capture_manager startCaptureWithDescriptor:descriptor error:&error])
                    throw std::runtime_error([[error description] UTF8String]);
            }
            auto* input = static_cast<float*>([x contents]);
            auto* weight = static_cast<float*>([w contents]);
            auto* result = static_cast<float*>([y contents]);
            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
            for (size_t i = 0; i < count; ++i) input[i] = distribution(rng);
            for (uint32_t i = 0; i < columns; ++i) weight[i] = distribution(rng);
            std::fill(result, result + count + 2 * guard, 123456.0f);
            auto dispatch = [&](int repeats) {
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                if (!command || !encoder) throw std::runtime_error("Metal encoding failed");
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:x offset:0 atIndex:0];
                [encoder setBuffer:w offset:0 atIndex:1];
                [encoder setBuffer:y offset:guard * sizeof(float) atIndex:2];
                [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
                [encoder setBytes:&columns length:sizeof(columns) atIndex:4];
                const NSUInteger threads = std::min<NSUInteger>(128, pipeline.maxTotalThreadsPerThreadgroup);
                for (int i = 0; i < repeats; ++i)
                    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted)
                    throw std::runtime_error([[command.error description] UTF8String]);
                return (command.GPUEndTime - command.GPUStartTime) * 1000.0 / repeats;
            };
            dispatch(1);
            double latency = 0;
            if (bench) {
                for (int i = 0; i < 3; ++i) dispatch(20);
                std::vector<double> timings;
                for (int i = 0; i < 7; ++i) timings.push_back(dispatch(20));
                std::sort(timings.begin(), timings.end());
                latency = timings[timings.size() / 2];
                if (!std::isfinite(latency) || latency <= 0) throw std::runtime_error("invalid GPU timestamp");
            }
            double max_error = 0;
            for (uint32_t row = 0; row < rows; ++row) {
                double squares = 0;
                for (uint32_t col = 0; col < columns; ++col) {
                    const double value = input[size_t(row) * columns + col];
                    squares += value * value;
                }
                const double scale = 1.0 / std::sqrt(squares / columns + 1e-5);
                for (uint32_t col = 0; col < columns; ++col) {
                    const size_t index = size_t(row) * columns + col;
                    const double expected = input[index] * scale * weight[col];
                    const double difference = std::abs(result[guard + index] - expected);
                    if (!std::isfinite(result[guard + index]) || difference > 2e-5 + 2e-5 * std::abs(expected))
                        throw std::runtime_error("CPU reference mismatch");
                    max_error = std::max(max_error, difference);
                }
            }
            for (size_t i = 0; i < guard; ++i)
                if (result[i] != 123456.0f || result[guard + count + i] != 123456.0f)
                    throw std::runtime_error("output guard overwritten");
            if (capture_manager) [capture_manager stopCapture];
            NSDictionary* report = @{@"invalid": capture.empty() ? @[] : @[@"profile capture enabled"],
                                     @"latency_ms": @(latency), @"max_abs_error": @(max_error),
                                     @"rows": @(rows), @"columns": @(columns), @"seed": @(seed),
                                     @"device": device.name, @"correct": @YES};
            NSData* data = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:&error];
            if (!data) throw std::runtime_error("JSON encoding failed");
            std::string text(static_cast<const char*>(data.bytes), data.length);
            if (!output.empty()) {
                std::ofstream file(output);
                file << text << '\n';
                if (!file) throw std::runtime_error("cannot write output JSON");
            }
            std::cout << text << '\n';
            return 0;
        } catch (const std::exception& error) {
            if (capture_manager && capture_manager.isCapturing) [capture_manager stopCapture];
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
}
