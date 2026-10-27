// Sightline AI — Client Controller & Assistive Interaction Layer

let currentJobId = null;
let jobData = null;
let narrations = [];
let activeNarrationIndex = -1;
let isAudioDucking = false;
let recognition = null;

// DOM Elements
const srStatus = document.getElementById("sr-status");
const srAlert = document.getElementById("sr-alert");
const fileInput = document.getElementById("video-file-input");
const btnSample = document.getElementById("btn-sample-video");
const btnStartAnalysis = document.getElementById("btn-start-analysis");
const btnDeleteJob = document.getElementById("btn-delete-job");
const uploadStatusMsg = document.getElementById("upload-status-msg");
const progressContainer = document.getElementById("progress-container");
const progressBarFill = document.getElementById("progress-bar-fill");
const progressText = document.getElementById("progress-text");
const playbackSection = document.getElementById("playback-section");
const mainVideo = document.getElementById("main-video");
const narrationAudio = document.getElementById("narration-audio");
const btnPlayPause = document.getElementById("btn-play-pause");
const playPauseText = document.getElementById("play-pause-text");
const playPauseIcon = document.getElementById("play-pause-icon");
const btnSeekBack = document.getElementById("btn-seek-back");
const btnSeekFwd = document.getElementById("btn-seek-fwd");
const btnRepeatLast = document.getElementById("btn-repeat-last");
const btnMoreDetail = document.getElementById("btn-more-detail");
const btnAskQuestion = document.getElementById("btn-ask-question");
const activeNarrationText = document.getElementById("active-narration-text");
const transcriptList = document.getElementById("transcript-list");
const qaForm = document.getElementById("qa-form");
const qaInput = document.getElementById("qa-input");
const btnVoiceInput = document.getElementById("btn-voice-input");
const qaResultBox = document.getElementById("qa-result-box");
const qaAnswerText = document.getElementById("qa-answer-text");
const sliderPace = document.getElementById("slider-pace");
const paceDisplay = document.getElementById("pace-display");
const btnThemeToggle = document.getElementById("btn-theme-toggle");
const btnAudioGuide = document.getElementById("btn-audio-guide");
const btnShortcutsHelp = document.getElementById("btn-shortcuts-help");
const shortcutsDialog = document.getElementById("shortcuts-dialog");
const btnCloseShortcuts = document.getElementById("btn-close-shortcuts");
const inputGeminiKey = document.getElementById("input-gemini-key");

// Restore saved Gemini API key if present
if (localStorage.getItem("sightline_gemini_key") && inputGeminiKey) {
  inputGeminiKey.value = localStorage.getItem("sightline_gemini_key");
}

// Announce to Screen Readers
function announceSR(text, isAlert = false) {
  if (isAlert) {
    srAlert.textContent = "";
    setTimeout(() => { srAlert.textContent = text; }, 50);
  } else {
    srStatus.textContent = "";
    setTimeout(() => { srStatus.textContent = text; }, 50);
  }
}

// Speak via Browser Web Speech API (Fallback / Orientation)
function speakText(text) {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = parseFloat(sliderPace.value);
    window.speechSynthesis.speak(utterance);
  }
}

// Audio-First Onboarding Guide
function playOnboardingGuide() {
  const guide = "Welcome to Sightline AI. You can operate this entire application using your keyboard. " +
    "Press Space to play or pause video and narration. Press R to repeat the last spoken description. " +
    "Press D to request more scene detail, or Q to ask a question at any moment. " +
    "Press Question Mark at any time to open the full list of keyboard shortcuts.";
  announceSR(guide, true);
  speakText(guide);
}

// On initial load, offer spoken audio guide
window.addEventListener("DOMContentLoaded", () => {
  announceSR("Sightline AI loaded. Press Space or Tab to explore controls, or click Audio Guide for orientation.");
});

btnAudioGuide.addEventListener("click", playOnboardingGuide);

// Theme Toggle
btnThemeToggle.addEventListener("click", () => {
  const isHighContrast = document.body.classList.toggle("theme-high-contrast");
  btnThemeToggle.setAttribute("aria-pressed", isHighContrast);
  announceSR(isHighContrast ? "High contrast mode enabled." : "Standard dark theme enabled.");
});

// Shortcuts Dialog
btnShortcutsHelp.addEventListener("click", () => {
  shortcutsDialog.showModal();
  btnCloseShortcuts.focus();
});
btnCloseShortcuts.addEventListener("click", () => {
  shortcutsDialog.close();
  btnShortcutsHelp.focus();
});

// Slider Pace
sliderPace.addEventListener("input", (e) => {
  paceDisplay.textContent = `${e.target.value}x`;
  announceSR(`Speech speed set to ${e.target.value} times normal.`);
});

// File Selection & Upload
fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  await handleUpload(file);
});

async function handleUpload(file) {
  uploadStatusMsg.textContent = `Uploading ${file.name}...`;
  announceSR(`Uploading ${file.name}...`);
  btnStartAnalysis.disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Upload failed.");
    }
    const data = await res.json();
    currentJobId = data.job_id;
    jobData = data;

    uploadStatusMsg.textContent = `Video ready: ${data.filename} (${data.duration}s, ${data.resolution}).`;
    announceSR(`Video uploaded successfully. Duration ${data.duration} seconds. Press Start AI Video Analysis.`);
    btnStartAnalysis.disabled = false;
    btnStartAnalysis.focus();
  } catch (err) {
    uploadStatusMsg.textContent = `Error: ${err.message}`;
    announceSR(`Upload error: ${err.message}`, true);
  }
}

// Sample Test Video Loader
btnSample.addEventListener("click", async () => {
  uploadStatusMsg.textContent = "Loading built-in sample test video...";
  announceSR("Loading built-in sample test video...");

  try {
    const res = await fetch("/api/sample-video");
    if (!res.ok) {
      throw new Error("Sample video endpoint not ready. Please select a video file.");
    }
    const data = await res.json();
    currentJobId = data.job_id;
    jobData = data;
    uploadStatusMsg.textContent = `Loaded sample: ${data.filename} (${data.duration}s).`;
    announceSR(`Sample video loaded. Press Start AI Video Analysis.`);
    btnStartAnalysis.disabled = false;
    btnStartAnalysis.focus();
  } catch (err) {
    uploadStatusMsg.textContent = err.message;
    announceSR(err.message, true);
  }
});

// Start Analysis
btnStartAnalysis.addEventListener("click", async () => {
  if (!currentJobId) return;

  btnStartAnalysis.disabled = true;
  progressContainer.style.display = "block";
  progressText.textContent = "Initiating video analysis pipeline...";
  announceSR("Starting AI video analysis. Please wait...");

  const mode = document.getElementById("select-narration-mode").value;
  const detail = document.getElementById("select-detail-level").value;
  const voice = document.getElementById("select-voice").value;
  const pace = parseFloat(sliderPace.value);

  const formData = new FormData();
  formData.append("mode", mode);
  formData.append("detail_level", detail);
  formData.append("voice", voice);
  formData.append("pace", pace);

  if (inputGeminiKey && inputGeminiKey.value.trim()) {
    const gKey = inputGeminiKey.value.trim();
    formData.append("gemini_key", gKey);
    localStorage.setItem("sightline_gemini_key", gKey);
  }

  try {
    const res = await fetch(`/api/process/${currentJobId}`, { method: "POST", body: formData });
    if (!res.ok) throw new Error("Could not start processing.");
    pollJobStatus();
  } catch (err) {
    progressText.textContent = `Error: ${err.message}`;
    announceSR(`Analysis failed: ${err.message}`, true);
    btnStartAnalysis.disabled = false;
  }
});

// Poll Status
async function pollJobStatus() {
  const interval = setInterval(async () => {
    try {
      const res = await fetch(`/api/status/${currentJobId}`);
      if (!res.ok) return;
      const data = await res.json();
      jobData = data;

      progressBarFill.style.width = `${data.progress}%`;
      progressText.textContent = `${data.progress}%: ${data.message}`;

      if (data.status === "ready") {
        clearInterval(interval);
        announceSR("Analysis complete! Synchronized spoken video experience is ready.", true);
        initPlayback(data);
      } else if (data.status === "failed") {
        clearInterval(interval);
        progressText.textContent = `Failed: ${data.error || "Unknown error"}`;
        announceSR(`Processing failed: ${data.error}`, true);
        btnStartAnalysis.disabled = false;
      }
    } catch (e) {
      clearInterval(interval);
    }
  }, 1200);
}

// Initialize Playback
function initPlayback(data) {
  progressContainer.style.display = "none";
  playbackSection.style.display = "block";
  btnDeleteJob.style.display = "inline-flex";

  const headingElem = document.getElementById("active-narration-heading");
  if (headingElem && data.vision_provider) {
    headingElem.textContent = `Current Spoken Description (${data.vision_provider}):`;
  }

  mainVideo.src = `/api/video/${data.job_id}`;
  narrations = data.narrations || [];

  // Populate Transcript
  transcriptList.innerHTML = "";
  narrations.forEach((item, index) => {
    const li = document.createElement("li");
    li.className = "transcript-item";
    li.tabIndex = 0;
    li.role = "button";
    li.setAttribute("aria-label", `Timestamp ${item.timestamp} seconds: ${item.text}`);
    li.innerHTML = `
      <span class="transcript-time">${formatTime(item.timestamp)}</span>
      <span class="transcript-text">${item.text}</span>
    `;
    li.addEventListener("click", () => jumpToNarration(index));
    li.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        jumpToNarration(index);
      }
    });
    transcriptList.appendChild(li);
  });

  activeNarrationIndex = -1;
  activeNarrationText.textContent = "Ready. Press Space to play video and hear narration.";
  btnPlayPause.focus();
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}

// Play / Pause Coordination
function togglePlayPause() {
  if (!mainVideo.src) return;
  if (mainVideo.paused) {
    mainVideo.play();
    playPauseIcon.textContent = "⏸️";
    playPauseText.textContent = "Pause (Space)";
    announceSR("Playing video and narration.");
  } else {
    mainVideo.pause();
    narrationAudio.pause();
    playPauseIcon.textContent = "▶️";
    playPauseText.textContent = "Play (Space)";
    announceSR("Paused.");
  }
}

btnPlayPause.addEventListener("click", togglePlayPause);

// Seeking
btnSeekBack.addEventListener("click", () => {
  mainVideo.currentTime = Math.max(0, mainVideo.currentTime - 5);
  announceSR(`Jumped backward to ${Math.round(mainVideo.currentTime)} seconds.`);
});

btnSeekFwd.addEventListener("click", () => {
  mainVideo.currentTime = Math.min(mainVideo.duration || 9999, mainVideo.currentTime + 5);
  announceSR(`Jumped forward to ${Math.round(mainVideo.currentTime)} seconds.`);
});

// Jump to specific narration
function jumpToNarration(index) {
  const item = narrations[index];
  if (!item) return;
  mainVideo.currentTime = item.timestamp;
  playNarration(index);
  if (mainVideo.paused) mainVideo.play();
}

// Video Timeupdate & Narration Triggering
mainVideo.addEventListener("timeupdate", () => {
  const curTime = mainVideo.currentTime;

  // Find next narration item scheduled near current time
  for (let i = 0; i < narrations.length; i++) {
    const item = narrations[i];
    if (Math.abs(curTime - item.timestamp) < 0.6 && activeNarrationIndex !== i) {
      playNarration(i);
      break;
    }
  }
});

// Play Narration & Handle Audio Ducking
function playNarration(index) {
  activeNarrationIndex = index;
  const item = narrations[index];
  if (!item) return;

  activeNarrationText.textContent = item.text;
  announceSR(item.text);

  // Update Transcript active item
  const allItems = transcriptList.querySelectorAll(".transcript-item");
  allItems.forEach((el, i) => {
    if (i === index) {
      el.setAttribute("aria-current", "true");
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else {
      el.removeAttribute("aria-current");
    }
  });

  // Audio Ducking: Reduce video volume while narration speaks
  mainVideo.volume = 0.15;
  isAudioDucking = true;

  if (item.audio_base64) {
    narrationAudio.src = `data:audio/wav;base64,${item.audio_base64}`;
    narrationAudio.playbackRate = parseFloat(sliderPace.value);
    narrationAudio.play().catch(() => speakText(item.text));
  } else if (item.audio_url) {
    narrationAudio.src = item.audio_url;
    narrationAudio.playbackRate = parseFloat(sliderPace.value);
    narrationAudio.play().catch(() => speakText(item.text));
  } else {
    speakText(item.text);
    setTimeout(restoreAudioDucking, item.duration_estimate * 1000);
  }
}

narrationAudio.addEventListener("ended", restoreAudioDucking);

function restoreAudioDucking() {
  if (isAudioDucking) {
    mainVideo.volume = 1.0;
    isAudioDucking = false;
  }
}

// Repeat Last Narration (R)
function repeatLastNarration() {
  if (activeNarrationIndex >= 0 && narrations[activeNarrationIndex]) {
    playNarration(activeNarrationIndex);
    announceSR(`Repeating: ${narrations[activeNarrationIndex].text}`);
  } else if (narrations.length > 0) {
    playNarration(0);
  }
}

btnRepeatLast.addEventListener("click", repeatLastNarration);

// Request More Detail (D)
async function requestMoreDetail() {
  if (!currentJobId) return;
  announceSR("Requesting more detailed scene description...", true);
  try {
    const res = await fetch(`/api/qa/${currentJobId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: "Describe this scene in more detail, including object positions, actions, and environmental context.",
        current_time: mainVideo.currentTime
      })
    });
    const data = await res.json();
    activeNarrationText.textContent = data.answer;
    announceSR(`Detailed description: ${data.answer}`, true);
    if (data.audio_base64) {
      mainVideo.volume = 0.15;
      narrationAudio.src = `data:audio/wav;base64,${data.audio_base64}`;
      narrationAudio.play();
    } else {
      speakText(data.answer);
    }
  } catch (e) {
    announceSR("Could not fetch detailed description.", true);
  }
}

btnMoreDetail.addEventListener("click", requestMoreDetail);

// Interactive Q&A (Q)
btnAskQuestion.addEventListener("click", () => {
  qaInput.focus();
  announceSR("Question input focused. Type your question and press Enter, or use voice.");
});

qaForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = qaInput.value.trim();
  if (!query || !currentJobId) return;

  qaResultBox.style.display = "block";
  qaAnswerText.textContent = "Analyzing video evidence...";
  announceSR("Analyzing video evidence for your question...", true);

  try {
    const res = await fetch(`/api/qa/${currentJobId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: query,
        current_time: mainVideo.currentTime
      })
    });
    const data = await res.json();
    qaAnswerText.textContent = data.answer;
    announceSR(`Answer: ${data.answer}`, true);

    if (data.audio_base64) {
      mainVideo.volume = 0.15;
      narrationAudio.src = `data:audio/wav;base64,${data.audio_base64}`;
      narrationAudio.play();
    } else {
      speakText(data.answer);
    }
  } catch (err) {
    qaAnswerText.textContent = "Could not obtain answer.";
    announceSR("Error answering question.", true);
  }
});

// Voice Input via Web Speech Recognition API
if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRec();
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    qaInput.value = transcript;
    announceSR(`Recognized question: ${transcript}`);
    qaForm.dispatchEvent(new Event("submit"));
  };

  recognition.onerror = () => {
    announceSR("Voice input not recognized. Please type your question.", true);
  };
}

btnVoiceInput.addEventListener("click", () => {
  if (recognition) {
    recognition.start();
    announceSR("Listening... Please speak your question now.");
  } else {
    announceSR("Voice recognition not supported in this browser. Please type.", true);
  }
});

// Delete Video & Data
btnDeleteJob.addEventListener("click", async () => {
  if (!currentJobId) return;
  if (!confirm("Are you sure you want to permanently delete this video and all associated narration data?")) return;

  try {
    await fetch(`/api/job/${currentJobId}`, { method: "DELETE" });
    announceSR("All data and media have been permanently deleted.", true);
    currentJobId = null;
    jobData = null;
    playbackSection.style.display = "none";
    btnDeleteJob.style.display = "none";
    btnStartAnalysis.disabled = true;
    fileInput.value = "";
    uploadStatusMsg.textContent = "Data deleted. Ready for new video.";
  } catch (e) {
    announceSR("Failed to delete job data.", true);
  }
});

// Global Keyboard Shortcut Controller
document.addEventListener("keydown", (e) => {
  // If typing in input or textarea, don't trigger global navigation keys
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) {
    if (e.key === "Escape") {
      document.activeElement.blur();
    }
    return;
  }

  switch (e.key.toLowerCase()) {
    case " ":
    case "k":
      e.preventDefault();
      togglePlayPause();
      break;
    case "r":
      e.preventDefault();
      repeatLastNarration();
      break;
    case "d":
      e.preventDefault();
      requestMoreDetail();
      break;
    case "q":
      e.preventDefault();
      qaInput.focus();
      break;
    case "j":
    case "arrowleft":
      e.preventDefault();
      btnSeekBack.click();
      break;
    case "l":
    case "arrowright":
      e.preventDefault();
      btnSeekFwd.click();
      break;
    case "[":
      e.preventDefault();
      sliderPace.value = Math.max(0.75, parseFloat(sliderPace.value) - 0.25);
      sliderPace.dispatchEvent(new Event("input"));
      break;
    case "]":
      e.preventDefault();
      sliderPace.value = Math.min(1.75, parseFloat(sliderPace.value) + 0.25);
      sliderPace.dispatchEvent(new Event("input"));
      break;
    case "m":
      e.preventDefault();
      mainVideo.muted = !mainVideo.muted;
      announceSR(mainVideo.muted ? "Video audio muted." : "Video audio unmuted.");
      break;
    case "?":
      e.preventDefault();
      shortcutsDialog.showModal();
      break;
    case "escape":
      if (shortcutsDialog.open) {
        shortcutsDialog.close();
      } else {
        if ("speechSynthesis" in window) window.speechSynthesis.cancel();
        narrationAudio.pause();
        restoreAudioDucking();
      }
      break;
  }
});
