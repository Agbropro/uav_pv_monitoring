const API_BASE = "/api/v1";
const PAGE_SIZE = 10;
const element = (id) => document.getElementById(id);
let offset = 0;
let hasMore = false;
let historyLoading = false;
let previewUrl = null;
let filters = new URLSearchParams();

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new Error("Cannot reach the backend. Check that the API is running, then try again.");
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    const message = Array.isArray(detail)
      ? detail.map((item) => `${item.loc?.slice(1).join(".") || "Request"}: ${item.msg}`).join("; ")
      : typeof detail === "string" ? detail : `Request failed (${response.status}).`;
    throw new Error(message);
  }
  return response.json();
}

function setMessage(id, text, error = false) {
  element(id).textContent = text;
  element(id).classList.toggle("error", error);
}

function anomalyStatus(category) {
  return category === "No-Anomaly" ? "No anomaly detected" : "Anomaly detected";
}

function formatTime(value) {
  const utcValue = /(?:Z|[+-]\d{2}:\d{2})$/i.test(value) ? value : `${value}Z`;
  return new Date(utcValue).toLocaleString();
}

function confidenceText(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function showInspection(inspection) {
  const prediction = inspection.classification;
  const image = inspection.inspection_image;
  element("result-placeholder").hidden = true;
  element("result").hidden = false;
  element("result-status").textContent = anomalyStatus(prediction.fault_type);
  element("result-status").className = `badge ${prediction.fault_type === "No-Anomaly" ? "normal" : "anomaly"}`;
  element("result-category").textContent = prediction.fault_type;
  element("result-confidence").textContent = confidenceText(prediction.confidence);
  element("result-asset").textContent = inspection.asset_id;
  element("result-observed").textContent = formatTime(image.timestamp);
  element("result-coordinates").textContent = `${image.latitude}, ${image.longitude}`;
  element("result-json").textContent = JSON.stringify(inspection, null, 2);
  const resultImage = element("result-image");
  resultImage.hidden = false;
  element("image-message").textContent = "";
  resultImage.onerror = () => {
    resultImage.hidden = true;
    element("image-message").textContent = "The saved image could not be loaded.";
  };
  resultImage.src = `${API_BASE}/inspections/${encodeURIComponent(inspection.inspection_id)}/image`;
}

element("image").addEventListener("change", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const file = element("image").files[0];
  element("preview").hidden = !file;
  if (file) {
    previewUrl = URL.createObjectURL(file);
    element("preview-image").src = previewUrl;
    element("preview-name").textContent = file.name;
  }
});

element("inspection-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = element("submit-button");
  button.disabled = true;
  button.textContent = "Inspecting…";
  setMessage("upload-message", "Uploading and running the model…");
  try {
    const data = new FormData(event.currentTarget);
    data.set("observed_at", new Date(element("observed-at").value).toISOString());
    const inspection = await request("/inspections", { method: "POST", body: data });
    showInspection(inspection);
    setMessage("upload-message", "Inspection saved.");
    offset = 0;
    await loadHistory();
  } catch (error) {
    setMessage("upload-message", error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "Run inspection";
  }
});

function updatePagination() {
  element("previous").disabled = historyLoading || offset === 0;
  element("next").disabled = historyLoading || !hasMore;
  element("refresh").disabled = historyLoading;
  for (const control of element("filter-form").elements) control.disabled = historyLoading;
}

async function loadHistory() {
  if (historyLoading) return;
  historyLoading = true;
  updatePagination();
  element("history-rows").replaceChildren();
  setMessage("history-message", "Loading inspections…");
  try {
    const query = new URLSearchParams(filters);
    query.set("limit", PAGE_SIZE);
    query.set("offset", offset);
    const page = await request(`/inspections?${query}`);
    hasMore = page.has_more;
    for (const inspection of page.items) {
      const row = document.createElement("tr");
      const prediction = inspection.classification;
      for (const value of [formatTime(inspection.timestamp), inspection.asset_id,
        anomalyStatus(prediction.fault_type), prediction.fault_type, confidenceText(prediction.confidence)]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      const cell = document.createElement("td");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "secondary";
      button.textContent = "View";
      button.setAttribute("aria-label", `View inspection for ${inspection.asset_id} at ${formatTime(inspection.timestamp)}`);
      button.addEventListener("click", () => {
        showInspection(inspection);
        element("result-heading").scrollIntoView({ behavior: "smooth", block: "start" });
      });
      cell.append(button);
      row.append(cell);
      element("history-rows").append(row);
    }
    setMessage("history-message", page.items.length ? "" : "No inspections match these filters.");
    element("page-label").textContent = page.items.length
      ? `Records ${offset + 1}–${offset + page.items.length} · newest first` : "No records on this page";
  } catch (error) {
    hasMore = false;
    element("page-label").textContent = "History unavailable";
    setMessage("history-message", error.message, true);
  } finally {
    historyLoading = false;
    updatePagination();
  }
}

element("filter-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const after = element("filter-after").value;
  const before = element("filter-before").value;
  if (after && before && new Date(after) > new Date(before)) {
    setMessage("history-message", "The start date must be before the end date.", true);
    return;
  }
  filters = new URLSearchParams();
  for (const [key, id] of [["asset_id", "filter-asset"], ["fault_type", "filter-fault"]]) {
    const value = element(id).value.trim();
    if (value) filters.set(key, value);
  }
  if (after) filters.set("inspected_after", new Date(after).toISOString());
  if (before) filters.set("inspected_before", new Date(before).toISOString());
  offset = 0;
  loadHistory();
});

element("clear-filters").addEventListener("click", () => {
  element("filter-form").reset();
  filters = new URLSearchParams();
  offset = 0;
  loadHistory();
});
element("refresh").addEventListener("click", () => loadHistory());
element("previous").addEventListener("click", () => { offset = Math.max(0, offset - PAGE_SIZE); loadHistory(); });
element("next").addEventListener("click", () => { offset += PAGE_SIZE; loadHistory(); });

const now = new Date();
element("observed-at").value = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
loadHistory();
