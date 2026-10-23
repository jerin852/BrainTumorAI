
import { useEffect, useState } from "react";
import {
  Brain,
  Upload,
  LayoutDashboard,
  ScanSearch,
  ChartNoAxesCombined,
  Activity,
  FileImage,
  ArrowRight,
  LoaderCircle,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  ShieldAlert,
  Eye,
  Info,
} from "lucide-react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";
const API_URL = `${API_BASE_URL}/predict`;

const tumorClasses = [
  { key: "glioma", label: "Glioma" },
  { key: "meningioma", label: "Meningioma" },
  { key: "notumor", label: "No Tumor" },
  { key: "pituitary", label: "Pituitary" },
];

const displayNames = {
  densenet121: "DenseNet121",
  convnext_tiny: "ConvNeXt-Tiny",
  efficientnetv2_s: "EfficientNetV2-S",
};

function normalizeEvaluationData(data) {
  const records = Array.isArray(data)
    ? data
    : Array.isArray(data?.models)
      ? data.models
      : [];

  return records.map((item) => {
    const key = String(
      item.model_name || item.model || item.name || "unknown"
    ).toLowerCase();

    const rawAccuracy = Number(
      item.test_accuracy ?? item.accuracy ?? item.accuracy_percent ?? 0
    );

    const accuracy =
      Number.isFinite(rawAccuracy) && rawAccuracy <= 1
        ? rawAccuracy * 100
        : Number.isFinite(rawAccuracy)
          ? rawAccuracy
          : 0;

    const report = item.classification_report || {};
    const macro = report["macro avg"] || {};
    const weighted = report["weighted avg"] || {};

    return {
      key,
      name: displayNames[key] || item.model_name || item.model || item.name || "Unknown model",
      accuracy,
      macroPrecision: Number(item.macro_precision ?? macro.precision ?? 0),
      macroRecall: Number(item.macro_recall ?? macro.recall ?? 0),
      macroF1: Number(item.macro_f1 ?? macro["f1-score"] ?? 0),
      weightedPrecision: Number(weighted.precision ?? 0),
      weightedRecall: Number(weighted.recall ?? 0),
      weightedF1: Number(weighted["f1-score"] ?? 0),
      testImages: Number(item.test_images ?? item.total_test_images ?? 0),
      classNames: item.class_names || [],
      confusionMatrix: item.confusion_matrix || [],
      classificationReport: report,
      status: key === "convnext_tiny" ? "Active API model" : "Evaluated",
      details: item,
    };
  });
}

function percent(value) {
  const number = Number(value);
  return `${(Number.isFinite(number) ? number * 100 : 0).toFixed(2)}%`;
}

function classLabel(name) {
  return tumorClasses.find((item) => item.key === name)?.label || name;
}

function MetricCard({ label, value, description }) {
  return (
    <article className="model-card">
      <p style={{ color: "#94a3b8", fontSize: 12, marginBottom: 8 }}>
        {label}
      </p>
      <h2 style={{ fontSize: 25, margin: "0 0 8px" }}>{value}</h2>
      {description && (
        <p style={{ color: "#94a3b8", fontSize: 12 }}>{description}</p>
      )}
    </article>
  );
}

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState("");
  const [page, setPage] = useState("Dashboard");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [apiStatus, setApiStatus] = useState("Checking backend...");
  const [heatmapOpacity, setHeatmapOpacity] = useState(75);

  const [modelMetrics, setModelMetrics] = useState([]);
  const [evaluationLoading, setEvaluationLoading] = useState(true);
  const [evaluationError, setEvaluationError] = useState("");
  const [selectedModelKey, setSelectedModelKey] = useState("convnext_tiny");

  // Image preview URL lifecycle.
  useEffect(() => {
    if (!image) {
      setPreview("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(image);
    setPreview(objectUrl);

    return () => URL.revokeObjectURL(objectUrl);
  }, [image]);

  // Backend health check.
  useEffect(() => {
    let cancelled = false;

    async function checkBackend() {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) throw new Error("Health check failed");

        if (!cancelled) setApiStatus("Backend connected");
      } catch {
        if (!cancelled) setApiStatus("Backend unavailable");
      }
    }

    checkBackend();

    return () => {
      cancelled = true;
    };
  }, []);

  // Load saved evaluation JSON from FastAPI.
  useEffect(() => {
    let cancelled = false;

    async function loadEvaluation() {
      setEvaluationLoading(true);
      setEvaluationError("");

      try {
        const response = await fetch(`${API_BASE_URL}/evaluation`);

        if (!response.ok) {
          let message = `Evaluation request failed (${response.status}).`;

          try {
            const body = await response.json();
            if (body.detail) {
              message =
                typeof body.detail === "string"
                  ? body.detail
                  : JSON.stringify(body.detail);
            }
          } catch {
            // Use the default error message.
          }

          throw new Error(message);
        }

        const data = await response.json();
        const models = normalizeEvaluationData(data);

        if (models.length === 0) {
          throw new Error(
            "No model evaluation records were returned by the API."
          );
        }

        if (!cancelled) {
          setModelMetrics(models);
          setSelectedModelKey((current) =>
            models.some((model) => model.key === current)
              ? current
              : models.find((model) => model.key === "convnext_tiny")?.key ||
                models[0].key
          );
        }
      } catch (err) {
        if (!cancelled) {
          setEvaluationError(
            err instanceof TypeError
              ? "Cannot connect to the evaluation API. Make sure FastAPI is running at http://127.0.0.1:8000."
              : err.message || "Unable to load evaluation results."
          );
        }
      } finally {
        if (!cancelled) setEvaluationLoading(false);
      }
    }

    loadEvaluation();

    return () => {
      cancelled = true;
    };
  }, []);

  function handleImage(event) {
    const file = event.target.files?.[0];
    event.target.value = "";

    if (!file) return;

    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      setError("Please select a JPG, PNG, or WebP image.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setError("Image size must be 10 MB or less.");
      return;
    }

    setImage(file);
    setResult(null);
    setError("");
    setHeatmapOpacity(75);
  }

  async function handleAnalyze() {
    if (!image) {
      setError("Please select a brain MRI image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", image);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let message = `Prediction request failed (${response.status}).`;

        try {
          const body = await response.json();
          if (body.detail) {
            message =
              typeof body.detail === "string"
                ? body.detail
                : JSON.stringify(body.detail);
          }
        } catch {
          // Keep the default message.
        }

        throw new Error(message);
      }

      const data = await response.json();

      if (
        !data.predicted_class ||
        typeof data.confidence_percent !== "number"
      ) {
        throw new Error("The API returned an unexpected prediction response.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err instanceof TypeError
          ? "Cannot connect to FastAPI. Make sure the backend is running at http://127.0.0.1:8000."
          : err.message || "Prediction failed. Please try again."
      );
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setImage(null);
    setResult(null);
    setError("");
    setLoading(false);
    setHeatmapOpacity(75);
  }

  const menu = [
    { name: "Dashboard", icon: LayoutDashboard },
    { name: "MRI Analysis", icon: ScanSearch },
    { name: "Model Evaluation", icon: ChartNoAxesCombined },
  ];

  const predictedClassKey = result?.predicted_class?.toLowerCase() || "";
  const predictedLabel = result
    ? classLabel(predictedClassKey)
    : "";

  const confidence = result
    ? Math.min(100, Math.max(0, result.confidence_percent))
    : 0;

  const probabilities = result?.probabilities || {};

  const bestModel = modelMetrics.reduce(
    (best, current) =>
      !best || current.accuracy > best.accuracy ? current : best,
    null
  );

  const totalTestImageEvaluations = modelMetrics.reduce(
    (total, model) => total + model.testImages,
    0
  );

  const selectedEvaluationModel =
    modelMetrics.find((model) => model.key === selectedModelKey) ||
    modelMetrics[0] ||
    null;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Brain size={25} />
          </div>
          <div>
            <h2>NeuroVision</h2>
            <span>AI RESEARCH PLATFORM</span>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>

        <nav>
          {menu.map(({ name, icon: Icon }) => (
            <button
              type="button"
              key={name}
              className={`nav-item ${page === name ? "active" : ""}`}
              onClick={() => setPage(name)}
            >
              <Icon size={19} />
              <span>{name}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div
            className="status-dot"
            style={{
              background:
                apiStatus === "Backend connected"
                  ? "#34d399"
                  : apiStatus === "Backend unavailable"
                    ? "#f87171"
                    : "#fbbf24",
            }}
          />
          <span>{apiStatus}</span>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">BRAIN TUMOR CLASSIFICATION</p>
            <h1>{page}</h1>
          </div>

          <div className="model-badge">
            <Activity size={17} />
            <span>AI Workspace</span>
          </div>
        </header>

        {/* DASHBOARD */}
        {page === "Dashboard" && (
          <>
            <section className="welcome">
              <div>
                <p className="eyebrow light">DEEP LEARNING RESEARCH</p>
                <h2>
                  Understand MRI.
                  <br />
                  Explore AI predictions.
                </h2>
                <p>
                  A research platform for brain MRI classification, model
                  evaluation, and visual exploration of model predictions.
                </p>

                <button
                  type="button"
                  className="primary-button"
                  onClick={() => setPage("MRI Analysis")}
                >
                  Start MRI analysis <ArrowRight size={18} />
                </button>
              </div>

              <div className="brain-art">
                <Brain size={116} strokeWidth={1} />
              </div>
            </section>

            <section className="section-heading">
              <div>
                <h2>Evaluation overview</h2>
                <p>Metrics loaded from the FastAPI evaluation endpoint</p>
              </div>
            </section>

            <section className="model-grid">
              <article className="model-card">
                <span className="model-number">01</span>
                <h3>Best test accuracy</h3>
                {evaluationLoading ? (
                  <p>Loading...</p>
                ) : bestModel ? (
                  <>
                    <h2>{bestModel.accuracy.toFixed(2)}%</h2>
                    <p>{bestModel.name}</p>
                  </>
                ) : (
                  <p>Evaluation data unavailable</p>
                )}
              </article>

              <article className="model-card highlighted">
                <span className="model-number">02</span>
                <h3>Models evaluated</h3>
                <h2>{evaluationLoading ? "..." : modelMetrics.length}</h2>
                <p>Models returned by the evaluation API</p>
                <span className="tag">MODEL EVALUATION</span>
              </article>

              <article className="model-card">
                <span className="model-number">03</span>
                <h3>Test-image evaluations</h3>
                <h2>
                  {evaluationLoading
                    ? "..."
                    : totalTestImageEvaluations.toLocaleString()}
                </h2>
                <p>Sum across model evaluation records</p>
              </article>
            </section>

            {evaluationError && (
              <div className="notice" role="alert">
                <AlertCircle size={20} />
                <p>{evaluationError}</p>
              </div>
            )}

            <section className="section-heading">
              <div>
                <h2>Research models</h2>
                <p>Saved test-set results for each architecture</p>
              </div>

              <button
                type="button"
                className="primary-button"
                onClick={() => setPage("Model Evaluation")}
              >
                Advanced metrics <ArrowRight size={17} />
              </button>
            </section>

            {evaluationLoading && (
              <div className="notice" aria-live="polite">
                <LoaderCircle
                  size={20}
                  style={{ animation: "spin 1s linear infinite" }}
                />
                <p>Loading model evaluation results...</p>
              </div>
            )}

            {!evaluationLoading && !evaluationError && (
              <section className="model-grid">
                {modelMetrics.map((model, index) => (
                  <article
                    className={`model-card ${
                      model.key === "convnext_tiny" ? "highlighted" : ""
                    }`}
                    key={model.key}
                  >
                    <span className="model-number">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <h3>{model.name}</h3>
                    <p>
                      Test accuracy: <strong>{model.accuracy.toFixed(2)}%</strong>
                    </p>
                    <p>
                      Macro F1: <strong>{model.macroF1.toFixed(4)}</strong>
                    </p>
                    <span className="tag">{model.status.toUpperCase()}</span>
                  </article>
                ))}
              </section>
            )}

            <div className="notice">
              <ShieldAlert size={20} />
              <p>
                <strong>Research status:</strong> ConvNeXt-Tiny is the active
                prediction model. Saved test-set metrics do not guarantee
                individual predictions or establish clinical validity.
              </p>
            </div>
          </>
        )}

        {/* MRI ANALYSIS */}
        {page === "MRI Analysis" && (
          <section className="analysis-layout">
            <article className="panel upload-panel">
              <div className="panel-title">
                <FileImage size={21} />
                <div>
                  <h2>MRI image input</h2>
                  <p>Select an image to preview it.</p>
                </div>
              </div>

              <label className="upload-area">
                {preview ? (
                  <img src={preview} alt="Selected brain MRI preview" />
                ) : (
                  <>
                    <Upload size={34} />
                    <strong>Upload brain MRI</strong>
                    <span>Click to select an image file</span>
                    <small>JPG, PNG or WebP · Maximum 10 MB</small>
                  </>
                )}

                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleImage}
                  hidden
                />
              </label>

              {image && (
                <div className="file-info">
                  <FileImage size={18} />
                  <span>{image.name}</span>
                </div>
              )}

              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 10,
                  marginTop: 18,
                }}
              >
                <button
                  type="button"
                  className="primary-button"
                  onClick={handleAnalyze}
                  disabled={!image || loading}
                  style={{
                    opacity: !image || loading ? 0.6 : 1,
                    cursor: !image || loading ? "not-allowed" : "pointer",
                  }}
                >
                  {loading ? (
                    <>
                      <LoaderCircle
                        size={18}
                        style={{ animation: "spin 1s linear infinite" }}
                      />
                      Analyzing MRI...
                    </>
                  ) : (
                    <>
                      <ScanSearch size={18} />
                      Analyze MRI
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={handleReset}
                  disabled={loading}
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "11px 16px",
                    borderRadius: 10,
                    border: "1px solid #334155",
                    background: "#172033",
                    color: "#dbeafe",
                    cursor: loading ? "not-allowed" : "pointer",
                  }}
                >
                  <RefreshCw size={16} />
                  Reset
                </button>
              </div>

              {error && (
                <div
                  role="alert"
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    gap: 8,
                    color: "#fca5a5",
                    background: "rgba(127,29,29,0.2)",
                    border: "1px solid rgba(248,113,113,0.35)",
                    padding: 12,
                    borderRadius: 10,
                    marginTop: 16,
                    overflowWrap: "anywhere",
                  }}
                >
                  <AlertCircle size={19} />
                  <span>{error}</span>
                </div>
              )}
            </article>

            <article className="panel result-panel">
              <div className="panel-title">
                <ScanSearch size={21} />
                <div>
                  <h2>Analysis result</h2>
                  <p>
                    {loading
                      ? "Processing image with ConvNeXt-Tiny..."
                      : result
                        ? "Prediction completed"
                        : "Upload an MRI and start analysis"}
                  </p>
                </div>
              </div>

              {loading && (
                <div className="empty-result" aria-live="polite">
                  <LoaderCircle
                    size={42}
                    style={{ animation: "spin 1s linear infinite" }}
                  />
                  <h3>Analyzing MRI</h3>
                  <p>
                    Processing the image and generating a Grad-CAM visualization.
                  </p>
                </div>
              )}

              {!loading && !result && (
                <div className="empty-result">
                  <Brain size={42} />
                  <h3>No prediction available</h3>
                  <p>
                    Select an MRI image and click Analyze MRI to request a
                    prediction from the connected backend.
                  </p>
                </div>
              )}

              {!loading && result && (
                <div style={{ marginTop: 22 }}>
                  <div
                    style={{
                      padding: 18,
                      borderRadius: 14,
                      background: "rgba(59,130,246,0.10)",
                      border: "1px solid rgba(96,165,250,0.3)",
                    }}
                  >
                    <p
                      style={{
                        color: "#93c5fd",
                        fontSize: 12,
                        letterSpacing: 1.2,
                        margin: "0 0 8px",
                      }}
                    >
                      MODEL PREDICTION
                    </p>

                    <h2
                      style={{
                        fontSize: 27,
                        margin: "0 0 12px",
                        overflowWrap: "anywhere",
                      }}
                    >
                      {predictedLabel}
                    </h2>

                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 12,
                        color: "#cbd5e1",
                      }}
                    >
                      <span>Model confidence score</span>
                      <strong>{confidence.toFixed(2)}%</strong>
                    </div>

                    <div
                      style={{
                        height: 8,
                        borderRadius: 10,
                        background: "#263449",
                        overflow: "hidden",
                        marginTop: 10,
                      }}
                    >
                      <div
                        style={{
                          height: "100%",
                          width: `${confidence}%`,
                          background: "#60a5fa",
                          borderRadius: 10,
                          transition: "width 0.4s ease",
                        }}
                      />
                    </div>
                  </div>

                  <div
                    role="note"
                    style={{
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 10,
                      marginTop: 16,
                      padding: 14,
                      borderRadius: 10,
                      background: "rgba(251,191,36,0.08)",
                      border: "1px solid rgba(251,191,36,0.3)",
                      color: "#fde68a",
                      fontSize: 13,
                      lineHeight: 1.6,
                    }}
                  >
                    <ShieldAlert size={21} style={{ flexShrink: 0 }} />
                    <div>
                      <strong>Prediction reliability notice</strong>
                      <p style={{ margin: "6px 0 0", color: "#cbd5e1" }}>
                        This score reflects the model output and is not
                        necessarily a calibrated probability of correctness.
                        AI predictions can be wrong and must not be used alone
                        for medical decisions.
                      </p>
                    </div>
                  </div>

                  <div style={{ marginTop: 24 }}>
                    <h3 style={{ marginBottom: 14 }}>Class probabilities</h3>

                    {tumorClasses.map(({ key, label }) => {
                      const raw = Number(probabilities[key] ?? 0);
                      const probability = Number.isFinite(raw)
                        ? Math.min(100, Math.max(0, raw))
                        : 0;

                      return (
                        <div key={key} style={{ marginBottom: 15 }}>
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              gap: 10,
                              marginBottom: 6,
                              color: "#cbd5e1",
                              fontSize: 13,
                            }}
                          >
                            <span>{label}</span>
                            <strong>{probability.toFixed(2)}%</strong>
                          </div>

                          <div
                            style={{
                              height: 7,
                              background: "#263449",
                              borderRadius: 10,
                              overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                height: "100%",
                                width: `${probability}%`,
                                background:
                                  key === predictedClassKey ? "#60a5fa" : "#64748b",
                                borderRadius: 10,
                                transition: "width 0.4s ease",
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {result.gradcam_image && preview && (
                    <section style={{ marginTop: 28 }}>
                      <div className="panel-title">
                        <Eye size={21} />
                        <div>
                          <h3>Explainable AI visualization</h3>
                          <p>
                            Compare the original MRI with the Grad-CAM output.
                          </p>
                        </div>
                      </div>

                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns:
                            "repeat(auto-fit, minmax(min(100%, 220px), 1fr))",
                          gap: 14,
                          marginTop: 16,
                        }}
                      >
                        {[
                          {
                            title: "Original MRI",
                            src: preview,
                            alt: "Original uploaded brain MRI",
                          },
                          {
                            title: "Grad-CAM heatmap",
                            src: result.gradcam_image,
                            alt: "Grad-CAM visualization",
                          },
                        ].map((item) => (
                          <div
                            key={item.title}
                            style={{
                              padding: 12,
                              borderRadius: 12,
                              background: "#101827",
                              border: "1px solid #334155",
                              minWidth: 0,
                            }}
                          >
                            <h4 style={{ margin: "0 0 12px" }}>{item.title}</h4>
                            <img
                              src={item.src}
                              alt={item.alt}
                              style={{
                                display: "block",
                                width: "100%",
                                height: 230,
                                objectFit: "contain",
                                background: "#080d18",
                                borderRadius: 8,
                              }}
                            />
                          </div>
                        ))}
                      </div>

                      <div
                        style={{
                          marginTop: 18,
                          padding: 16,
                          border: "1px solid #334155",
                          borderRadius: 12,
                          background: "#101827",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: 12,
                            marginBottom: 10,
                          }}
                        >
                          <label
                            htmlFor="heatmap-opacity"
                            style={{ color: "#e2e8f0", fontSize: 13 }}
                          >
                            Grad-CAM display opacity
                          </label>
                          <strong style={{ color: "#93c5fd", fontSize: 13 }}>
                            {heatmapOpacity}%
                          </strong>
                        </div>

                        <input
                          id="heatmap-opacity"
                          type="range"
                          min="10"
                          max="100"
                          step="5"
                          value={heatmapOpacity}
                          onChange={(event) =>
                            setHeatmapOpacity(Number(event.target.value))
                          }
                          style={{
                            width: "100%",
                            accentColor: "#60a5fa",
                            cursor: "pointer",
                          }}
                        />

                        <p
                          style={{
                            color: "#94a3b8",
                            fontSize: 12,
                            margin: "8px 0 0",
                            lineHeight: 1.6,
                          }}
                        >
                          Changes the displayed heatmap opacity only. It does
                          not recalculate Grad-CAM or change the prediction.
                        </p>

                        <div style={{ marginTop: 16 }}>
                          <h4 style={{ margin: "0 0 10px" }}>
                            Adjustable Grad-CAM view
                          </h4>

                          <div
                            style={{
                              position: "relative",
                              width: "100%",
                              minHeight: 180,
                              overflow: "hidden",
                              borderRadius: 10,
                              background: "#080d18",
                              border: "1px solid #334155",
                            }}
                          >
                            <img
                              src={preview}
                              alt="Original MRI underneath Grad-CAM"
                              style={{
                                display: "block",
                                width: "100%",
                                height: 300,
                                objectFit: "contain",
                              }}
                            />

                            <img
                              src={result.gradcam_image}
                              alt="Grad-CAM overlay with adjustable opacity"
                              style={{
                                position: "absolute",
                                inset: 0,
                                display: "block",
                                width: "100%",
                                height: "100%",
                                objectFit: "contain",
                                opacity: heatmapOpacity / 100,
                                transition: "opacity 0.15s ease",
                              }}
                            />
                          </div>
                        </div>
                      </div>

                      <div
                        style={{
                          display: "flex",
                          alignItems: "flex-start",
                          gap: 9,
                          color: "#94a3b8",
                          fontSize: 12,
                          lineHeight: 1.6,
                          marginTop: 14,
                        }}
                      >
                        <Info size={18} style={{ flexShrink: 0 }} />
                        <p style={{ margin: 0 }}>
                          Grad-CAM highlights regions associated with the
                          model output. It does not prove that a tumor exists
                          or that a highlighted region is medically significant.
                        </p>
                      </div>
                    </section>
                  )}

                  {!result.gradcam_image && (
                    <div className="notice" style={{ marginTop: 20 }}>
                      <Info size={20} />
                      <p>
                        Grad-CAM was not returned by the backend for this
                        prediction. Check the backend response and logs.
                      </p>
                    </div>
                  )}

                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      color: "#86efac",
                      marginTop: 20,
                      fontSize: 13,
                    }}
                  >
                    <CheckCircle2 size={18} />
                    Prediction received successfully
                  </div>

                  <p
                    style={{
                      color: "#94a3b8",
                      fontSize: 12,
                      marginTop: 14,
                      overflowWrap: "anywhere",
                    }}
                  >
                    File: {result.filename || image?.name}
                    <br />
                    Model: {result.model || "ConvNeXt-Tiny"}
                  </p>
                </div>
              )}

              <p className="disclaimer">
                Research demonstration only. This output is not a medical
                diagnosis. Model confidence does not establish clinical certainty.
              </p>
            </article>
          </section>
        )}

        {/* ADVANCED MODEL EVALUATION */}
        {page === "Model Evaluation" && (
          <section className="panel evaluation-panel">
            <div className="panel-title">
              <ChartNoAxesCombined size={22} />
              <div>
                <h2>Advanced Model Evaluation</h2>
                <p>
                  Per-class metrics and confusion matrix from saved test results.
                </p>
              </div>
            </div>

            {evaluationLoading && (
              <div className="empty-result" aria-live="polite">
                <LoaderCircle
                  size={32}
                  style={{ animation: "spin 1s linear infinite" }}
                />
                <h3>Loading evaluation results</h3>
                <p>Fetching saved metrics from FastAPI.</p>
              </div>
            )}

            {evaluationError && (
              <div className="notice" role="alert">
                <AlertCircle size={20} />
                <p>{evaluationError}</p>
              </div>
            )}

            {!evaluationLoading &&
              !evaluationError &&
              selectedEvaluationModel && (
                <>
                  <div style={{ marginTop: 22 }}>
                    <label
                      htmlFor="evaluation-model"
                      style={{
                        display: "block",
                        marginBottom: 9,
                        color: "#cbd5e1",
                        fontSize: 13,
                      }}
                    >
                      Select model to inspect
                    </label>

                    <select
                      id="evaluation-model"
                      value={selectedEvaluationModel.key}
                      onChange={(event) =>
                        setSelectedModelKey(event.target.value)
                      }
                      style={{
                        width: "100%",
                        maxWidth: 420,
                        padding: "12px 14px",
                        borderRadius: 10,
                        border: "1px solid #334155",
                        background: "#101827",
                        color: "#e2e8f0",
                        fontSize: 14,
                      }}
                    >
                      {modelMetrics.map((model) => (
                        <option key={model.key} value={model.key}>
                          {model.name}
                          {model.key === "convnext_tiny"
                            ? " — Active API model"
                            : ""}
                        </option>
                      ))}
                    </select>
                  </div>

                  <h3 style={{ marginTop: 28, marginBottom: 14 }}>
                    {selectedEvaluationModel.name} — Summary metrics
                  </h3>

                  <section
                    className="model-grid"
                    style={{ marginTop: 0 }}
                  >
                    <MetricCard
                      label="Test accuracy"
                      value={`${selectedEvaluationModel.accuracy.toFixed(2)}%`}
                      description={`${selectedEvaluationModel.testImages.toLocaleString()} test images`}
                    />
                    <MetricCard
                      label="Macro Precision"
                      value={percent(selectedEvaluationModel.macroPrecision)}
                    />
                    <MetricCard
                      label="Macro Recall"
                      value={percent(selectedEvaluationModel.macroRecall)}
                    />
                    <MetricCard
                      label="Macro F1 Score"
                      value={percent(selectedEvaluationModel.macroF1)}
                    />
                    <MetricCard
                      label="Weighted Precision"
                      value={percent(selectedEvaluationModel.weightedPrecision)}
                    />
                    <MetricCard
                      label="Weighted Recall"
                      value={percent(selectedEvaluationModel.weightedRecall)}
                    />
                    <MetricCard
                      label="Weighted F1 Score"
                      value={percent(selectedEvaluationModel.weightedF1)}
                    />
                  </section>

                  {/* PER-CLASS METRICS */}
                  <h3 style={{ marginTop: 32, marginBottom: 12 }}>
                    Per-class performance
                  </h3>

                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Class</th>
                          <th>Precision</th>
                          <th>Recall</th>
                          <th>F1 Score</th>
                          <th>Support</th>
                        </tr>
                      </thead>

                      <tbody>
                        {selectedEvaluationModel.classNames.map((name) => {
                          const report =
                            selectedEvaluationModel.classificationReport[name] ||
                            {};

                          return (
                            <tr key={name}>
                              <td>{classLabel(name)}</td>
                              <td>{percent(report.precision)}</td>
                              <td>{percent(report.recall)}</td>
                              <td>{percent(report["f1-score"])}</td>
                              <td>
                                {Number(report.support || 0).toLocaleString()}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* CONFUSION MATRIX */}
                  <h3 style={{ marginTop: 32, marginBottom: 8 }}>
                    Confusion Matrix
                  </h3>

                  <p
                    style={{
                      color: "#94a3b8",
                      fontSize: 13,
                      lineHeight: 1.6,
                    }}
                  >
                    Rows represent actual classes; columns represent predicted
                    classes. Diagonal cells are correct predictions.
                  </p>

                  {selectedEvaluationModel.confusionMatrix.length > 0 ? (
                    <div
                      style={{
                        overflowX: "auto",
                        marginTop: 14,
                        border: "1px solid #334155",
                        borderRadius: 12,
                      }}
                    >
                      <table style={{ minWidth: 540, width: "100%" }}>
                        <thead>
                          <tr>
                            <th>Actual ↓ / Predicted →</th>
                            {selectedEvaluationModel.classNames.map((name) => (
                              <th key={name}>{classLabel(name)}</th>
                            ))}
                          </tr>
                        </thead>

                        <tbody>
                          {selectedEvaluationModel.confusionMatrix.map(
                            (row, rowIndex) => (
                              <tr
                                key={
                                  selectedEvaluationModel.classNames[rowIndex] ||
                                  rowIndex
                                }
                              >
                                <th>
                                  {classLabel(
                                    selectedEvaluationModel.classNames[rowIndex]
                                  )}
                                </th>

                                {row.map((count, colIndex) => {
                                  const diagonal = rowIndex === colIndex;
                                  const rowTotal = row.reduce(
                                    (sum, value) => sum + Number(value || 0),
                                    0
                                  );
                                  const intensity =
                                    rowTotal > 0
                                      ? Math.min(
                                          1,
                                          Number(count || 0) / rowTotal
                                        )
                                      : 0;

                                  return (
                                    <td
                                      key={colIndex}
                                      style={{
                                        textAlign: "center",
                                        fontWeight: diagonal ? 700 : 500,
                                        background: diagonal
                                          ? `rgba(52,211,153,${0.12 + intensity * 0.48})`
                                          : Number(count) > 0
                                            ? `rgba(248,113,113,${0.08 + intensity * 0.35})`
                                            : "rgba(148,163,184,0.04)",
                                        color: diagonal ? "#d1fae5" : "#e2e8f0",
                                      }}
                                      title={`${count} test images`}
                                    >
                                      {count}
                                    </td>
                                  );
                                })}
                              </tr>
                            )
                          )}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="notice">
                      <Info size={20} />
                      <p>No confusion matrix was returned for this model.</p>
                    </div>
                  )}

                  {/* CLASS-SPECIFIC OBSERVATION */}
                  {selectedEvaluationModel.key === "convnext_tiny" &&
                    selectedEvaluationModel.classificationReport.glioma && (
                      <div className="notice" style={{ marginTop: 22 }}>
                        <ShieldAlert size={21} />
                        <p>
                          <strong>Research observation:</strong> ConvNeXt-Tiny
                          has a glioma recall of{" "}
                          {percent(
                            selectedEvaluationModel.classificationReport.glioma
                              .recall
                          )}{" "}
                          in this saved test report. Review the misclassified
                          images and class-specific results; overall metrics
                          can hide errors affecting individual classes.
                        </p>
                      </div>
                    )}

                  <div
                    style={{
                      marginTop: 18,
                      padding: 14,
                      borderRadius: 10,
                      background: "rgba(59,130,246,0.08)",
                      border: "1px solid rgba(96,165,250,0.2)",
                      color: "#cbd5e1",
                      fontSize: 13,
                      lineHeight: 1.7,
                    }}
                  >
                    <strong style={{ color: "#93c5fd" }}>
                      Evaluation methodology
                    </strong>
                    <p style={{ margin: "6px 0 0" }}>
                      Metrics are loaded from the saved test-set evaluation
                      report. This page does not retrain models or run a new
                      evaluation. Scores describe this dataset and evaluation
                      protocol; they do not establish clinical validity.
                    </p>
                  </div>
                </>
              )}

            <p className="disclaimer">
              Research use only. These results are not a medical diagnosis and
              must not be used alone to make clinical decisions.
            </p>
          </section>
        )}

        <footer>
          NeuroVision · Brain MRI AI Research · Experimental software
        </footer>
      </main>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        button:focus-visible,
        input:focus-visible,
        select:focus-visible {
          outline: 2px solid #60a5fa;
          outline-offset: 3px;
        }

        @media (max-width: 700px) {
          .analysis-layout {
            grid-template-columns: minmax(0, 1fr) !important;
          }
        }
      `}</style>
    </div>
  );
}

export default App;
