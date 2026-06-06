import CoreML
import Foundation
import AudioCommon

/// Kokoro-82M text-to-speech — CoreML-based, runs on Neural Engine.
///
/// Lightweight (82M params) non-autoregressive TTS model.
/// Supports 10 languages with 54 preset voices. Designed for iOS/iPad deployment.
///
/// Uses an end-to-end CoreML model (`kokoro_5s.mlmodelc`) that runs the full
/// pipeline (BERT → duration → alignment → prosody → decoder) in one call.
///
/// ```swift
/// let tts = try await KokoroTTSModel.fromPretrained()
/// let audio = try tts.synthesize(text: "Hello world", voice: "af_heart")
/// ```
public final class KokoroTTSModel {

    /// Default HuggingFace model ID.
    public static let defaultModelId = "aufklarer/Kokoro-82M-CoreML"
    /// Default voice preset.
    public static let defaultVoice = "af_heart"
    /// Output sample rate.
    public static let outputSampleRate = 24000

    let config: KokoroConfig
    var network: KokoroNetwork?
    let phonemizer: KokoroPhonemizer
    var voiceEmbeddings: [String: [Float]]

    var _isLoaded: Bool { network != nil }

    init(config: KokoroConfig, network: KokoroNetwork, phonemizer: KokoroPhonemizer, voiceEmbeddings: [String: [Float]]) {
        self.config = config
        self.network = network
        self.phonemizer = phonemizer
        self.voiceEmbeddings = voiceEmbeddings
    }

    // MARK: - Synthesis

    /// Synthesize speech from text.
    public func synthesize(
        text: String,
        voice: String = "af_heart",
        language: String = "en",
        speed: Float = 1.0
    ) throws -> [Float] {
        guard _isLoaded, let network else {
            throw AudioModelError.inferenceFailed(
                operation: "kokoro-synthesize", reason: "Model not loaded")
        }

        let tokenIds = phonemizer.tokenize(text, maxLength: config.maxPhonemeLength, language: language)
        let tokenCount = min(tokenIds.count, 128)

        guard let styleVector = voiceEmbeddings[voice] else {
            let available = Array(voiceEmbeddings.keys).sorted().prefix(5)
            throw AudioModelError.voiceNotFound(
                voice: voice,
                searchPath: "Available: \(available.joined(separator: ", "))...")
        }

        let padTo = 128
        let paddedIds = phonemizer.pad(Array(tokenIds.prefix(padTo)), to: padTo)

        let inputIds = try createInt32Array(shape: [1, padTo], values: paddedIds.map { Int32($0) })
        let maskArray = try createInt32Array(shape: [1, padTo], values: (0..<padTo).map { Int32($0 < tokenCount ? 1 : 0) })
        let refS = try createFloatArray(shape: [1, config.styleDim], values: styleVector)
        let speedArray = try createFloatArray(shape: [1], values: [speed])

        let t0 = CFAbsoluteTimeGetCurrent()
        let result = try network.predictE2E(inputIds: inputIds, attentionMask: maskArray, refS: refS, speed: speedArray)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        let validSamples = min(result.audioLengthSamples, result.audio.count)
        guard validSamples > 0 else { return [] }

        var audio = [Float](repeating: 0, count: validSamples)
        if result.audio.dataType == .float16 {
            let ptr = result.audio.dataPointer.bindMemory(to: Float16.self, capacity: validSamples)
            for i in 0..<validSamples { audio[i] = Float(ptr[i]) }
        } else {
            let ptr = result.audio.dataPointer.bindMemory(to: Float.self, capacity: validSamples)
            for i in 0..<validSamples { audio[i] = ptr[i] }
        }

        // Kokoro's E2E model often emits 100–300 ms of trailing artifacts
        // past the real speech (random low-energy noise + an occasional
        // loud-spike click, observed on iPhone playback). Find where real
        // speech actually ended by walking backwards through 10 ms windows
        // and locating the last window above a silence-energy threshold —
        // then zero everything past that, plus a short ramp-down on the
        // last few ms of real speech to avoid a click at the new boundary.
        // Use a 50 ms window with a higher silence floor and require the
        // window to be SUSTAINED above threshold — Kokoro's trailing
        // artifacts are often a single 10–20 ms spike with low surrounding
        // energy, so a tight short window otherwise mistakes them for
        // speech. Real speech tails consistently above ~0.03 RMS over a
        // 50 ms span; isolated artifact spikes don't.
        let win = max(1, Int(0.050 * Double(config.sampleRate)))
        let silenceRMS: Float = 0.030
        var speechEnd = validSamples
        var i = validSamples - win
        while i > 0 {
            var sumSq: Float = 0
            for j in 0..<win { let v = audio[i + j]; sumSq += v * v }
            let rms = sqrt(sumSq / Float(win))
            if rms > silenceRMS { speechEnd = i + win; break }
            i -= win / 2  // 50 % overlap so we don't miss a window boundary
        }
        // Zero the trailing artifact region.
        if speechEnd < validSamples {
            for k in speechEnd..<validSamples { audio[k] = 0 }
        }
        // Linear fade-out on the last ~10 ms of the kept signal so the
        // boundary between speech and the silenced region is also smooth.
        let fadeSamples = min(speechEnd, Int(0.010 * Double(config.sampleRate)))
        if fadeSamples >= 2 {
            let start = speechEnd - fadeSamples
            let denom = Float(fadeSamples - 1)
            for k in 0..<fadeSamples {
                let gain = Float(fadeSamples - 1 - k) / denom
                audio[start + k] *= gain
            }
        }

        let duration = Double(validSamples) / Double(config.sampleRate)
        let elapsedMs = elapsed * 1000
        AudioLog.inference.info("Kokoro E2E: \(tokenCount) tokens → \(validSamples) samples (\(String(format: "%.1f", duration))s) in \(String(format: "%.0f", elapsedMs))ms")
        return audio
    }

    /// List available voice presets.
    public var availableVoices: [String] {
        Array(voiceEmbeddings.keys).sorted()
    }

    // MARK: - Helpers

    private func createInt32Array(shape: [Int], values: [Int32]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { $0 as NSNumber }, dataType: .int32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<values.count { ptr[i] = values[i] }
        return arr
    }

    private func createFloatArray(shape: [Int], values: [Float]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { $0 as NSNumber }, dataType: .float32)
        let ptr = arr.dataPointer.assumingMemoryBound(to: Float.self)
        for i in 0..<values.count { ptr[i] = values[i] }
        return arr
    }

    // MARK: - Warmup

    /// Warm up CoreML model by running a dummy inference.
    public func warmUp() throws {
        _ = try? synthesize(text: "hello", voice: availableVoices.first ?? "af_heart")
    }

    // MARK: - Model Loading

    /// Load a pretrained Kokoro model from HuggingFace.
    ///
    /// - Parameter computeUnits: Which hardware the main CoreML model runs on.
    ///   Defaults to `.all` (Neural Engine preferred). Pass `.cpuAndGPU` to bypass
    ///   the Neural Engine — useful as a fallback on platforms where the ANE
    ///   compiler produces incorrect output for this model.
    public static func fromPretrained(
        modelId: String = defaultModelId,
        voice: String = defaultVoice,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> KokoroTTSModel {
        AudioLog.modelLoading.info("Loading Kokoro model: \(modelId)")

        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download E2E model + G2P + voice
        progressHandler?(0.0, "Downloading model...")
        try await HuggingFaceDownloader.downloadWeights(
            modelId: modelId,
            to: cacheDir,
            additionalFiles: [
                "kokoro_5s.mlmodelc/**",
                "G2PEncoder.mlmodelc/**",
                "G2PDecoder.mlmodelc/**",
                "vocab_index.json",
                "g2p_vocab.json",
                "us_gold.json",
                "us_silver.json",
                "voices/*.json",
            ],
            offlineMode: offlineMode
        ) { fraction in
            progressHandler?(fraction * 0.7, "Downloading model...")
        }

        return try loadFromDirectory(
            cacheDir,
            modelId: modelId,
            computeUnits: computeUnits,
            progressBase: 0.7,
            progressScale: 0.3,
            progressHandler: progressHandler
        )
    }

    /// Load a Kokoro model directly from a local directory.
    ///
    /// This loader never contacts HuggingFace and is intended for app-bundled or
    /// otherwise pre-downloaded model directories.
    ///
    /// Expected layout:
    ///
    /// ```text
    /// Kokoro/
    ///   kokoro_5s.mlmodelc/
    ///   G2PEncoder.mlmodelc/
    ///   G2PDecoder.mlmodelc/
    ///   vocab_index.json
    ///   g2p_vocab.json
    ///   us_gold.json
    ///   us_silver.json
    ///   voices/
    ///     af_heart.json
    /// ```
    public static func fromLocalDirectory(
        _ directory: URL,
        computeUnits: MLComputeUnits = .all,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> KokoroTTSModel {
        AudioLog.modelLoading.info("Loading local Kokoro model from \(directory.path)")

        progressHandler?(0.0, "Validating local model...")
        try validateLocalDirectory(directory)

        return try loadFromDirectory(
            directory,
            modelId: "kokoro-local",
            computeUnits: computeUnits,
            progressBase: 0.0,
            progressScale: 1.0,
            progressHandler: progressHandler
        )
    }

    private static func loadFromDirectory(
        _ directory: URL,
        modelId: String,
        computeUnits: MLComputeUnits,
        progressBase: Double,
        progressScale: Double,
        progressHandler: ((Double, String) -> Void)?
    ) throws -> KokoroTTSModel {
        func report(_ fraction: Double, _ status: String) {
            progressHandler?(progressBase + fraction * progressScale, status)
        }

        // Load vocabulary
        report(0.07, "Loading vocabulary...")
        let vocabURL = directory.appendingPathComponent("vocab_index.json")
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            throw AudioModelError.modelLoadFailed(modelId: modelId, reason: "vocab_index.json not found")
        }
        let phonemizer = try KokoroPhonemizer.loadVocab(from: vocabURL)
        try phonemizer.loadDictionaries(from: directory)

        // Load G2P models
        report(0.20, "Loading G2P models...")
        let g2pEncoderURL = directory.appendingPathComponent("G2PEncoder.mlmodelc", isDirectory: true)
        let g2pDecoderURL = directory.appendingPathComponent("G2PDecoder.mlmodelc", isDirectory: true)
        let g2pVocabURL = directory.appendingPathComponent("g2p_vocab.json")
        if FileManager.default.fileExists(atPath: g2pEncoderURL.path) &&
           FileManager.default.fileExists(atPath: g2pDecoderURL.path) {
            try phonemizer.loadG2PModels(
                encoderURL: g2pEncoderURL, decoderURL: g2pDecoderURL, vocabURL: g2pVocabURL)
            AudioLog.modelLoading.debug("Loaded CoreML G2P encoder + decoder")
        }

        // Load voice embeddings
        report(0.35, "Loading voice embeddings...")
        var voiceEmbeddings = [String: [Float]]()
        let voicesDir = directory.appendingPathComponent("voices")
        if FileManager.default.fileExists(atPath: voicesDir.path) {
            let files = try FileManager.default.contentsOfDirectory(at: voicesDir, includingPropertiesForKeys: nil)
            for file in files where file.pathExtension == "json" {
                let voiceName = file.deletingPathExtension().lastPathComponent
                if let embedding = try? loadVoiceEmbedding(from: file, styleDim: KokoroConfig.default.styleDim) {
                    voiceEmbeddings[voiceName] = embedding
                }
            }
            AudioLog.modelLoading.debug("Loaded \(voiceEmbeddings.count) voice presets")
        }

        guard !voiceEmbeddings.isEmpty else {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "No valid Kokoro voice embeddings found in \(voicesDir.path)")
        }

        // Load E2E CoreML model
        report(0.65, "Loading CoreML model...")
        let network = try KokoroNetwork(directory: directory, computeUnits: computeUnits)
        AudioLog.modelLoading.debug("Loaded Kokoro E2E model")

        report(1.0, "Model loaded")
        AudioLog.modelLoading.info("Kokoro model loaded successfully")

        return KokoroTTSModel(
            config: .default, network: network,
            phonemizer: phonemizer, voiceEmbeddings: voiceEmbeddings)
    }

    private static func validateLocalDirectory(_ directory: URL) throws {
        let fileManager = FileManager.default
        let modelId = "kokoro-local"

        func requireFile(_ name: String) throws {
            let url = directory.appendingPathComponent(name)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue else {
                throw AudioModelError.modelLoadFailed(
                    modelId: modelId,
                    reason: "Required file missing: \(url.path)")
            }
        }

        func requireDirectory(_ name: String) throws -> URL {
            let url = directory.appendingPathComponent(name, isDirectory: true)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                throw AudioModelError.modelLoadFailed(
                    modelId: modelId,
                    reason: "Required directory missing: \(url.path)")
            }
            return url
        }

        _ = try requireDirectory("kokoro_5s.mlmodelc")
        _ = try requireDirectory("G2PEncoder.mlmodelc")
        _ = try requireDirectory("G2PDecoder.mlmodelc")
        try requireFile("vocab_index.json")
        try requireFile("g2p_vocab.json")
        try requireFile("us_gold.json")
        try requireFile("us_silver.json")

        let voicesDir = try requireDirectory("voices")
        let voiceFiles = try fileManager.contentsOfDirectory(at: voicesDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
        guard !voiceFiles.isEmpty else {
            throw AudioModelError.modelLoadFailed(
                modelId: modelId,
                reason: "Required voice JSON missing: \(voicesDir.path) must contain at least one .json voice file")
        }
    }

    /// Load voice embedding from JSON file.
    private static func loadVoiceEmbedding(from url: URL, styleDim: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let embedding = json["embedding"] as? [Double] else { return [] }
        return embedding.prefix(styleDim).map { Float($0) }
    }
}
