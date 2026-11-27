// Recording a take and scanning the score with the device's camera (new file; gui/server.py inlines it into index.html).
//
// Where the browser can (getUserMedia, and MediaRecorder for video), a panel shows the camera: "Record" films the take with the microphone and puts
// it into the Recording field, "Scan" takes photos of the pages and puts them into the Score field. Where it cannot (older browsers, some phones'
// embedded views), the buttons open the device's own camera app through <input capture>, which every phone supports, and use what it returns.
// Both paths end the same way: a File set on the form's file input, so the upload and the analysis know nothing about the camera.
// A scanned page is cleaned for the music reader (Audiveris): shrunk to at most 2400 px, converted to grey and thresholded locally (Bradley's
// adaptive method over an integral image), which removes the shadows and uneven light of a phone photo; several pages become one PDF, built here.
const CAN_STREAM = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
const CAN_RECORD = CAN_STREAM && typeof MediaRecorder !== "undefined";
const PAGE_LONG_SIDE = 2400, THRESHOLD_SHARE = 0.12;

function giveFile(input, file) {                         // what the user picked, as the form's own file
  const transfer = new DataTransfer();
  transfer.items.add(file);
  input.files = transfer.files;
  input.dispatchEvent(new Event("change", {bubbles: true}));
}
async function startStream(video, constraints) {
  stopStream(video);
  video.srcObject = await navigator.mediaDevices.getUserMedia(constraints);
  await video.play();
}
function stopStream(video) {
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
}
const clock = (s) => Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");

// ---- cleaning a photographed page ---------------------------------------------------------------------------------------
function cleanPage(source) {                             // <video> or <img> -> canvas of black music on white paper
  const w0 = source.videoWidth || source.naturalWidth, h0 = source.videoHeight || source.naturalHeight;
  const k = Math.min(1, PAGE_LONG_SIDE / Math.max(w0, h0)), w = Math.round(w0 * k), h = Math.round(h0 * k);
  const canvas = document.createElement("canvas");
  canvas.width = w; canvas.height = h;
  const g = canvas.getContext("2d", {willReadFrequently: true});
  g.drawImage(source, 0, 0, w, h);
  const image = g.getImageData(0, 0, w, h), px = image.data, grey = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) grey[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
  const stride = w + 1, sum = new Float64Array(stride * (h + 1));       // integral image: the mean of any window in four lookups
  for (let y = 0; y < h; y++) {
    let row = 0;
    for (let x = 0; x < w; x++) { row += grey[y * w + x]; sum[(y + 1) * stride + x + 1] = sum[y * stride + x + 1] + row; }
  }
  const r = Math.max(8, Math.round(Math.max(w, h) / 32));               // window: about a thirty-second of the page on each side
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
      const mean = (sum[y1 * stride + x1] - sum[y0 * stride + x1] - sum[y1 * stride + x0] + sum[y0 * stride + x0]) / ((x1 - x0) * (y1 - y0));
      const v = grey[y * w + x] < mean * (1 - THRESHOLD_SHARE) ? 0 : 255;  // darker than its surroundings: ink
      px[4 * (y * w + x)] = px[4 * (y * w + x) + 1] = px[4 * (y * w + x) + 2] = v; px[4 * (y * w + x) + 3] = 255;
    }
  }
  g.putImageData(image, 0, 0);
  return canvas;
}
const toJpeg = (canvas) => new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));

function buildPdf(pages) {                               // pages: [{bytes: Uint8Array of a JPEG, w, h}] -> a PDF Blob, one image per page
  const enc = new TextEncoder(), chunks = [], offsets = [];
  let length = 0;
  const put = (part) => { const bytes = typeof part === "string" ? enc.encode(part) : part; chunks.push(bytes); length += bytes.length; };
  const object = (n, ...parts) => { offsets[n] = length; put(`${n} 0 obj\n`); parts.forEach(put); put("\nendobj\n"); };
  put("%PDF-1.4\n");
  const kids = pages.map((_, i) => `${3 + 3 * i} 0 R`).join(" ");
  object(1, "<< /Type /Catalog /Pages 2 0 R >>");
  object(2, `<< /Type /Pages /Kids [${kids}] /Count ${pages.length} >>`);
  pages.forEach((p, i) => {
    const page = 3 + 3 * i, content = page + 1, image = page + 2, draw = `q ${p.w} 0 0 ${p.h} 0 0 cm /Im0 Do Q`;
    object(page, `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${p.w} ${p.h}] /Resources << /XObject << /Im0 ${image} 0 R >> >> /Contents ${content} 0 R >>`);
    object(content, `<< /Length ${draw.length} >>\nstream\n${draw}\nendstream`);
    object(image, `<< /Type /XObject /Subtype /Image /Width ${p.w} /Height ${p.h} /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /DCTDecode /Length ${p.bytes.length} >>\nstream\n`, p.bytes, "\nendstream");
  });
  const count = 3 + 3 * pages.length, xref = length;
  put(`xref\n0 ${count}\n0000000000 65535 f \n`);
  for (let n = 1; n < count; n++) put(String(offsets[n]).padStart(10, "0") + " 00000 n \n");
  put(`trailer\n<< /Size ${count} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`);
  return new Blob(chunks, {type: "application/pdf"});
}

// ---- recording the take ------------------------------------------------------------------------------------------------
(function recorder() {
  const open = document.getElementById("rec-open"), panel = document.getElementById("rec-panel"), video = document.getElementById("rec-video"),
        go = document.getElementById("rec-go"), flip = document.getElementById("rec-flip"), cancel = document.getElementById("rec-cancel"),
        info = document.getElementById("rec-info"), target = document.querySelector("input[name=video]"), phone = document.getElementById("rec-phone");
  let facing = "user", recorder = null, chunks = [], started = 0, timer = null;
  const say = (text, bad) => { info.className = bad ? "error" : "hint"; info.textContent = text; };
  const close = () => { clearInterval(timer); if (recorder && recorder.state !== "inactive") recorder.onstop = null, recorder.stop(); recorder = null; stopStream(video); panel.hidden = true; go.textContent = "Start recording"; };
  const camera = () => startStream(video, {video: {facingMode: {ideal: facing}, width: {ideal: 1280}}, audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
  if (!CAN_RECORD) {                                     // the device's own camera app instead
    open.onclick = () => phone.click();
    phone.onchange = () => { if (phone.files[0]) { giveFile(target, phone.files[0]); say("Using the video from your camera."); } };
    return;
  }
  open.onclick = async () => {
    try { panel.hidden = false; await camera(); say(""); }
    catch (e) { close(); say("Could not use the camera and microphone: allow them for this page and try again (" + e.name + ").", true); }
  };
  flip.onclick = () => { facing = facing === "user" ? "environment" : "user"; camera().catch((e) => say("Could not switch the camera (" + e.name + ").", true)); };
  cancel.onclick = () => { close(); say(""); };
  go.onclick = () => {
    if (recorder) { recorder.stop(); return; }
    const type = ["video/mp4", "video/webm;codecs=vp9,opus", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t)) || "";
    chunks = [];
    recorder = new MediaRecorder(video.srcObject, type ? {mimeType: type} : undefined);
    recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      const seconds = (Date.now() - started) / 1000, mime = recorder.mimeType || type || "video/webm";
      const file = new File(chunks, "take." + (mime.includes("mp4") ? "mp4" : "webm"), {type: mime.split(";")[0]});
      close();
      giveFile(target, file);
      say(`Recorded a take of ${clock(seconds)} (${(file.size / 1e6).toFixed(1)} MB): it will be analysed.`);
    };
    recorder.start(1000);
    started = Date.now();
    go.textContent = "Stop recording";
    timer = setInterval(() => say("Recording " + clock((Date.now() - started) / 1000)), 250);
  };
})();

// ---- scanning the score ------------------------------------------------------------------------------------------------
(function scanner() {
  const open = document.getElementById("scan-open"), panel = document.getElementById("scan-panel"), video = document.getElementById("scan-video"),
        shot = document.getElementById("scan-shot"), done = document.getElementById("scan-done"), flip = document.getElementById("scan-flip"),
        cancel = document.getElementById("scan-cancel"), list = document.getElementById("scan-pages"), info = document.getElementById("scan-info"),
        target = document.querySelector("input[name=score]"), phone = document.getElementById("scan-phone");
  let facing = "environment";
  const pages = [];                                      // {bytes, w, h, url}
  const say = (text, bad) => { info.className = bad ? "error" : "hint"; info.textContent = text; };
  const camera = () => startStream(video, {video: {facingMode: {ideal: facing}, width: {ideal: 3840}, height: {ideal: 2160}}, audio: false});
  const redraw = () => {
    list.textContent = "";
    pages.forEach((p, i) => {
      const img = document.createElement("img");
      img.src = p.url; img.title = "Page " + (i + 1) + ": click to remove"; img.style.cssText = "height:90px;border:1px solid #bbb;cursor:pointer";
      img.onclick = () => { URL.revokeObjectURL(p.url); pages.splice(i, 1); redraw(); };
      list.append(img);
    });
    done.disabled = !pages.length;
    done.textContent = pages.length > 1 ? `Use ${pages.length} pages` : "Use this page";
  };
  const addPage = async (source) => {
    const canvas = cleanPage(source), blob = await toJpeg(canvas);
    pages.push({bytes: new Uint8Array(await blob.arrayBuffer()), w: canvas.width, h: canvas.height, url: URL.createObjectURL(blob), blob});
    redraw();
  };
  const finish = () => {
    const file = pages.length === 1 ? new File([pages[0].blob], "score.jpg", {type: "image/jpeg"})
                                    : new File([buildPdf(pages)], "score.pdf", {type: "application/pdf"});
    giveFile(target, file);
    say(`Scanned ${pages.length} page${pages.length > 1 ? "s" : ""}: reading a scan needs Audiveris on this computer (see the README).`);
    close();
  };
  const close = () => { stopStream(video); panel.hidden = true; pages.forEach((p) => URL.revokeObjectURL(p.url)); pages.length = 0; redraw(); };
  if (!CAN_STREAM) {                                     // the device's own camera app, one page at a time
    open.onclick = () => phone.click();
    phone.onchange = async () => {
      if (!phone.files[0]) return;
      const img = new Image();
      img.src = URL.createObjectURL(phone.files[0]);
      await img.decode();
      await addPage(img);
      finish();
    };
    return;
  }
  open.onclick = async () => {
    try { panel.hidden = false; await camera(); say(""); }
    catch (e) { close(); say("Could not use the camera: allow it for this page and try again (" + e.name + ").", true); }
  };
  flip.onclick = () => { facing = facing === "user" ? "environment" : "user"; camera().catch((e) => say("Could not switch the camera (" + e.name + ").", true)); };
  shot.onclick = () => addPage(video);
  done.onclick = finish;
  cancel.onclick = () => { close(); say(""); };
})();
