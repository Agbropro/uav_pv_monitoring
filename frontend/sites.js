const API_BASE = "/api/v1";
const PAGE_SIZE = 10;
const element = (id) => document.getElementById(id);
const pages = {
  sites: { offset: 0, hasMore: false, loading: false },
  assets: { offset: 0, hasMore: false, loading: false },
};
let selectedSite = null;

function updateButtons(kind) {
  const page = pages[kind];
  element(`${kind}-previous`).disabled = page.loading || page.offset === 0;
  element(`${kind}-next`).disabled = page.loading || !page.hasMore;
  element("refresh-sites").disabled = pages.sites.loading;
  element("all-assets").disabled = pages.assets.loading;
  for (const button of element("sites-rows").querySelectorAll("button")) {
    button.disabled = pages.assets.loading;
  }
}

function addCell(row, value) {
  const cell = document.createElement("td");
  cell.textContent = value;
  row.append(cell);
}

async function loadPage(kind) {
  const page = pages[kind];
  if (page.loading) return;
  page.loading = true;
  updateButtons(kind);
  const message = element(`${kind}-message`);
  message.classList.remove("error");
  message.textContent = `Loading ${kind}…`;
  element(`${kind}-rows`).replaceChildren();
  try {
    const query = new URLSearchParams({ limit: PAGE_SIZE, offset: page.offset });
    if (kind === "assets" && selectedSite !== null) query.set("site_id", selectedSite);
    const response = await fetch(`${API_BASE}/${kind}?${query}`);
    if (!response.ok) throw new Error(`Could not load ${kind} (${response.status}).`);
    const data = await response.json();
    page.hasMore = data.has_more;
    for (const item of data.items) {
      const row = document.createElement("tr");
      const values = kind === "sites"
        ? [item.site_id, item.name, item.latitude, item.longitude]
        : [item.asset_id, item.name, item.site_id, item.latitude, item.longitude];
      for (const value of values) addCell(row, value);
      if (kind === "sites") {
        const cell = document.createElement("td");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "secondary";
        button.textContent = "View assets";
        button.setAttribute("aria-label", `View assets for ${item.name}`);
        button.addEventListener("click", () => {
          if (pages.assets.loading) return;
          selectedSite = item.site_id;
          pages.assets.offset = 0;
          element("assets-heading").textContent = `Assets · ${item.name}`;
          loadPage("assets");
          element("assets-heading").scrollIntoView({ behavior: "smooth", block: "start" });
        });
        cell.append(button);
        row.append(cell);
      }
      element(`${kind}-rows`).append(row);
    }
    message.textContent = data.items.length ? "" : `No ${kind} found.`;
    element(`${kind}-page`).textContent = data.items.length
      ? `Records ${page.offset + 1}–${page.offset + data.items.length}` : "No records on this page";
  } catch (error) {
    page.hasMore = false;
    message.classList.add("error");
    message.textContent = error instanceof TypeError
      ? "Cannot reach the backend. Check that the API and database are running." : error.message;
    element(`${kind}-page`).textContent = "Records unavailable";
  } finally {
    page.loading = false;
    updateButtons(kind);
  }
}

for (const kind of ["sites", "assets"]) {
  element(`${kind}-previous`).addEventListener("click", () => {
    pages[kind].offset = Math.max(0, pages[kind].offset - PAGE_SIZE);
    loadPage(kind);
  });
  element(`${kind}-next`).addEventListener("click", () => {
    pages[kind].offset += PAGE_SIZE;
    loadPage(kind);
  });
}
element("refresh-sites").addEventListener("click", () => loadPage("sites"));
element("all-assets").addEventListener("click", () => {
  selectedSite = null;
  pages.assets.offset = 0;
  element("assets-heading").textContent = "All assets";
  loadPage("assets");
});
loadPage("sites");
loadPage("assets");
