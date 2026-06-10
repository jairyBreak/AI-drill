const state = {
  topology: null,
  snapshot: null,
  selected: null,
  topologyDrawn: false,
};

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 1) => Number.isFinite(Number(n)) ? Number(n).toFixed(d) : "-";
const pct = (n) => `${fmt(Number(n) * 100, 0)}%`;
const DEMO_NODES = new Set(["h1", "l1", "l2", "h2", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]);

function kindOf(id) {
  if (id.startsWith("h")) return "host";
  if (id.startsWith("s")) return "spine";
  return "leaf";
}

function layoutNodes(topology) {
  return {
    h2: { x: 600, y: 60 },
    l2: { x: 600, y: 165 },
    s1: { x: 145, y: 330 },
    s2: { x: 275, y: 330 },
    s3: { x: 405, y: 330 },
    s4: { x: 535, y: 330 },
    s5: { x: 665, y: 330 },
    s6: { x: 795, y: 330 },
    s7: { x: 925, y: 330 },
    s8: { x: 1055, y: 330 },
    l1: { x: 600, y: 500 },
    h1: { x: 600, y: 600 },
  };
}

function demoLinks() {
  return (state.topology?.links || []).filter((link) => {
    const pair = new Set([link.source, link.target]);
    if ([...pair].some((id) => !DEMO_NODES.has(id))) return false;
    if (pair.has("h1") && pair.has("l1")) return true;
    if (pair.has("h2") && pair.has("l2")) return true;
    if (pair.has("l1") && [...pair].some((id) => id.startsWith("s"))) return true;
    if (pair.has("l2") && [...pair].some((id) => id.startsWith("s"))) return true;
    return false;
  });
}

function portMetricFor(node, peer) {
  if (!state.snapshot || !state.snapshot.switches[node]) return null;
  const ports = state.snapshot.switches[node].ports || [];
  return ports.find((p) => p.neighbor === peer) || null;
}

function linkClass(link) {
  const a = portMetricFor(link.source, link.target);
  const b = portMetricFor(link.target, link.source);
  const q = Math.max(a?.queue_depth || 0, b?.queue_depth || 0);
  const m = Math.max(a?.mbps || 0, b?.mbps || 0);
  if (q > 40) return "bad active";
  if (q > 24) return "hot active";
  return m > 0.03 ? "active" : "";
}

function linkMetric(link) {
  const a = portMetricFor(link.source, link.target);
  const b = portMetricFor(link.target, link.source);
  return {
    mbps: Math.max(a?.mbps || 0, b?.mbps || 0),
    util: Math.max(a?.utilization || 0, b?.utilization || 0),
    queue: Math.max(a?.queue_depth || 0, b?.queue_depth || 0),
  };
}

function linkKey(link) {
  return `${link.source}-${link.target}`;
}

function offsetPoint(a, b, amount) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
  return { x: -dy / len * amount, y: dx / len * amount };
}

function linkInFocus(link) {
  if (!state.selected) return false;
  return link.source === state.selected || link.target === state.selected;
}

function updateTopologyDynamics() {
  if (!state.topologyDrawn || !state.topology) return;
  const pos = layoutNodes(state.topology);
  demoLinks().forEach((link) => {
    const key = linkKey(link);
    const metric = linkMetric(link);
    const focused = linkInFocus(link);
    const active = metric.mbps > 0.03;
    const width = 1.6 + Math.min(7, metric.mbps * 4.5);
    const line = document.querySelector(`.link[data-key="${key}"]`);
    if (line) {
      line.setAttribute("class", `link ${linkClass(link)}${state.selected && !focused ? " dim" : ""}${focused ? " focus" : ""}`);
      line.setAttribute("stroke-width", width.toFixed(1));
      const title = line.querySelector("title");
      if (title) {
        title.textContent = `${link.source} p${link.ports?.[link.source] ?? "-"} ↔ ${link.target} p${link.ports?.[link.target] ?? "-"} | bw ${link.bw || "-"}M | util ${pct(metric.util)} | now ${fmt(metric.mbps, 2)}M | q ${fmt(metric.queue, 0)}`;
      }
    }

    const badge = document.querySelector(`.traffic-badge[data-key="${key}"]`);
    if (badge) {
      const a = pos[link.source];
      const b = pos[link.target];
      const off = offsetPoint(a, b, 12);
      const midX = (a.x + b.x) / 2 + off.x;
      const midY = (a.y + b.y) / 2 + off.y;
      const show = active || metric.mbps > 0.08;
      badge.setAttribute("class", `traffic-badge${show ? "" : " hidden"}${state.selected && !focused ? " dim" : ""}${focused ? " focus" : ""}`);
      const rect = badge.querySelector("rect");
      const text = badge.querySelector("text");
      const w = focused ? 68 : 54;
      const h = focused ? 26 : 20;
      if (rect) {
        rect.setAttribute("x", midX - w / 2);
        rect.setAttribute("y", midY - h / 2);
        rect.setAttribute("width", w);
        rect.setAttribute("height", h);
      }
      if (text) {
        text.setAttribute("x", midX);
        text.setAttribute("y", midY + 1);
        text.textContent = `${fmt(metric.mbps, 2)}M`;
      }
    }
  });

  document.querySelectorAll(".node").forEach((node) => {
    const id = node.getAttribute("data-id");
    node.classList.toggle("selected", id === state.selected);
  });
}

function drawTopology() {
  if (!state.topology) return;
  const svg = $("topology");
  svg.innerHTML = "";
  const pos = layoutNodes(state.topology);

  const linksG = document.createElementNS("http://www.w3.org/2000/svg", "g");
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  defs.innerHTML = `
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" class="arrow-head"></path>
    </marker>
  `;
  svg.appendChild(defs);
  demoLinks().forEach((link) => {
    const a = pos[link.source];
    const b = pos[link.target];
    if (!a || !b) return;
    const metric = linkMetric(link);
    const focused = linkInFocus(link);
    const active = metric.mbps > 0.03;
    const width = 1.6 + Math.min(7, metric.mbps * 4.5);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    const key = linkKey(link);
    line.setAttribute("x1", a.x);
    line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x);
    line.setAttribute("y2", b.y);
    line.setAttribute("class", `link ${linkClass(link)}${state.selected && !focused ? " dim" : ""}${focused ? " focus" : ""}`);
    line.setAttribute("data-key", key);
    line.setAttribute("stroke-width", width.toFixed(1));
    if (link.source === "h1" || link.source === "l1" || link.target === "l2" || link.target === "h2") {
      line.setAttribute("marker-end", "url(#arrow)");
    }
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${link.source} p${link.ports?.[link.source] ?? "-"} ↔ ${link.target} p${link.ports?.[link.target] ?? "-"} | bw ${link.bw || "-"}M | util ${pct(metric.util)} | now ${fmt(metric.mbps, 2)}M | q ${fmt(metric.queue, 0)}`;
    line.appendChild(title);
    linksG.appendChild(line);

    const off = offsetPoint(a, b, 12);
    const midX = (a.x + b.x) / 2 + off.x;
    const midY = (a.y + b.y) / 2 + off.y;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", `traffic-badge hidden`);
    g.setAttribute("data-key", key);
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", midX - 27);
    rect.setAttribute("y", midY - 10);
    rect.setAttribute("width", 54);
    rect.setAttribute("height", 20);
    rect.setAttribute("rx", 5);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", midX);
    label.setAttribute("y", midY + 1);
    label.textContent = "0.00M";
    g.appendChild(rect);
    g.appendChild(label);
    linksG.appendChild(g);
  });
  svg.appendChild(linksG);

  const caption = document.createElementNS("http://www.w3.org/2000/svg", "text");
  caption.setAttribute("x", 600);
  caption.setAttribute("y", 638);
  caption.setAttribute("class", "flow-caption");
  caption.textContent = "Demo path: h1 -> l1 -> W-ECMP/DRILL spine fabric -> l2 -> h2";
  svg.appendChild(caption);

  const nodesG = document.createElementNS("http://www.w3.org/2000/svg", "g");
  state.topology.nodes.filter((node) => DEMO_NODES.has(node.id)).forEach((node) => {
    const p = pos[node.id];
    if (!p) return;
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    const kind = node.kind || kindOf(node.id);
    g.setAttribute("class", `node ${kind}${state.selected === node.id ? " selected" : ""}`);
    g.setAttribute("data-id", node.id);
    g.addEventListener("click", () => {
      if (kind !== "host") {
        state.selected = state.selected === node.id ? null : node.id;
        render();
      }
    });

    if (kind === "host") {
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", p.x);
      c.setAttribute("cy", p.y);
      c.setAttribute("r", 23);
      g.appendChild(c);
    } else {
      const frame = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      frame.setAttribute("class", "switch-frame");
      frame.setAttribute("x", p.x - 45);
      frame.setAttribute("y", p.y - 29);
      frame.setAttribute("width", 90);
      frame.setAttribute("height", 58);
      frame.setAttribute("rx", 7);
      g.appendChild(frame);

      const img = document.createElementNS("http://www.w3.org/2000/svg", "image");
      img.setAttribute("href", "/switch.svg");
      img.setAttribute("x", p.x - 39);
      img.setAttribute("y", p.y - 19);
      img.setAttribute("width", 78);
      img.setAttribute("height", 38);
      img.setAttribute("preserveAspectRatio", "xMidYMid meet");
      g.appendChild(img);
    }
    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", p.x);
    t.setAttribute("y", kind === "host" ? p.y : p.y + 43);
    t.textContent = node.id;
    g.appendChild(t);
    nodesG.appendChild(g);
  });
  svg.appendChild(nodesG);
  state.topologyDrawn = true;
  updateTopologyDynamics();
}

function updateTopMetrics() {
  const ml = state.snapshot?.ml || {};
  $("statusText").textContent = state.snapshot
    ? `${state.snapshot.mode || "-"} | ${state.snapshot.timestamp || "-"}`
    : "Connecting";
  $("predLat").textContent = `${fmt(ml.pred_latency_ms)} ms`;
  $("realLat").textContent = `${fmt(ml.real_latency_ms)} ms`;
  $("predLoss").textContent = `${fmt(ml.pred_loss_pct)}%`;
  $("realLoss").textContent = `${fmt(ml.real_loss_pct)}%`;
  $("totalMbps").textContent = `${fmt(ml.total_mbps, 2)} M`;
  $("rehashAge").textContent = `${fmt(ml.last_rehash_seconds, 0)} s`;
  const pill = $("modePill");
  pill.textContent = ml.anomaly ? "ANOMALY" : "NORMAL";
  pill.className = `pill ${ml.anomaly ? "bad" : ""}`;
}

function renderComponents() {
  const root = $("components");
  root.innerHTML = "";
  (state.snapshot?.components || []).forEach((c) => {
    const el = document.createElement("div");
    el.className = "component";
    const members = (c.members || []).map((m) => `${m.spine}:p${m.port}`).join("  ");
    el.innerHTML = `
      <div class="component-title">
        <strong>Component ${c.id}</strong>
        <span>${c.current_weight ?? "-"} / ${c.anchor_weight ?? "-"}</span>
      </div>
      <div class="members">${members}</div>
    `;
    root.appendChild(el);
  });
}

function queueBar(value, max = 64) {
  const ratio = Math.max(0, Math.min(1, Number(value) / max));
  const cls = value > 40 ? "bad" : value > 24 ? "warn" : "";
  return `<span class="bar"><span class="fill ${cls}" style="width:${ratio * 100}%"></span></span>${fmt(value, 0)}`;
}

function utilBar(value) {
  const ratio = Math.max(0, Math.min(1, Number(value)));
  const cls = value > 1.0 ? "bad" : value > 0.75 ? "warn" : "";
  return `<span class="bar"><span class="fill ${cls}" style="width:${ratio * 100}%"></span></span>${pct(value)}`;
}

function portUtil(port) {
  const cap = Number(port.effective_bw || 0);
  const mbps = Number(port.mbps || 0);
  return cap > 0 ? mbps / cap : 0;
}

function renderPorts() {
  const selected = "l2";
  const swMeta = state.topology?.switches?.[selected];
  const swData = state.snapshot?.switches?.[selected];
  $("detailTitle").textContent = "l2 Ports";
  $("detailMeta").textContent = swMeta
    ? `Thrift ${swMeta.thrift_port ?? "-"} | ${swData?.total_mbps ? fmt(swData.total_mbps, 2) : "0.00"} Mbps | util = Mbps / effective capacity | focus ${state.selected || "none"}`
    : "No switch selected";

  const rows = swData?.ports || swMeta?.ports || [];
  $("portsBody").innerHTML = rows.map((p) => `
    <tr>
      <td>${p.port}</td>
      <td>${p.neighbor || "-"}</td>
      <td>${utilBar(portUtil(p))}</td>
      <td>${fmt(p.mbps || 0, 2)}</td>
      <td>${fmt(p.queue_depth || 0, 0)}</td>
      <td>${fmt(p.ingress_pps || 0, 0)}</td>
      <td>${fmt(p.egress_pps || 0, 0)}</td>
    </tr>
  `).join("");
}

function render() {
  updateTopMetrics();
  if (!state.topologyDrawn) {
    drawTopology();
  } else {
    updateTopologyDynamics();
  }
  renderComponents();
  renderPorts();
}

async function loadTopology() {
  const res = await fetch("/api/topology", { cache: "no-store" });
  state.topology = await res.json();
  state.selected = null;
}

function connectEvents() {
  const es = new EventSource("/events");
  es.onmessage = (ev) => {
    state.snapshot = JSON.parse(ev.data);
    render();
  };
  es.onerror = () => {
    $("statusText").textContent = "Disconnected";
  };
}

(async function main() {
  await loadTopology();
  render();
  connectEvents();
})();
