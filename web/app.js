(() => {
  const lobby = document.getElementById('lobby');
  const lobbyTitle = document.getElementById('lobbyTitle');
  const lobbyCopy = document.getElementById('lobbyCopy');
  const lobbyPortraitSlot = document.getElementById('lobbyPortraitSlot');
  const stage = document.getElementById('stage');
  const done = document.getElementById('done');
  const doneDuration = document.getElementById('doneDuration');
  const doneQuestions = document.getElementById('doneQuestions');
  const controls = document.getElementById('controls');

  const avatar = document.getElementById('avatar');
  const statusPill = document.getElementById('statusPill');
  const statusLabel = document.getElementById('statusLabel');
  const lastLine = document.getElementById('lastLine');
  const transcript = document.getElementById('transcript');
  const transcriptEmpty = document.getElementById('transcriptEmpty');

  const headerProgress = document.getElementById('headerProgress');
  const progressLabel = document.getElementById('progressLabel');
  const progressFill = document.getElementById('progressFill');
  const elapsedEl = document.getElementById('elapsed');
  const connLabel = document.getElementById('connLabel');
  const connWrap = document.getElementById('conn');

  const micCheck = document.getElementById('micCheck');
  const micCheckLabel = document.getElementById('micCheckLabel');
  const micCheckSub = document.getElementById('micCheckSub');

  const testMicBtn = document.getElementById('testMicBtn');
  const micModalBackdrop = document.getElementById('micModalBackdrop');
  const micModalClose = document.getElementById('micModalClose');
  const modalMicDot = document.getElementById('modalMicDot');
  const modalMicStatus = document.getElementById('modalMicStatus');
  const modalMicDevice = document.getElementById('modalMicDevice');

  const endBtn = document.getElementById('endBtn');
  const endModalBackdrop = document.getElementById('endModalBackdrop');
  const endCancel = document.getElementById('endCancel');
  const endConfirm = document.getElementById('endConfirm');

  // Mirrors interview_agent.session.State exactly - see session.py.
  const STATE_LABELS = {
    idle: 'Getting started…',
    listening: 'Listening to you',
    user_speaking: 'You are speaking',
    processing: 'Processing your answer…',
    ai_thinking: 'Jerry is thinking…',
    ai_speaking: 'Jerry is speaking',
    interrupted: 'You cut in — Jerry is listening',
    error: 'Microphone issue detected',
  };

  const PILL_CLASS = {
    idle: '',
    listening: 'listening',
    user_speaking: 'user',
    processing: 'thinking',
    ai_thinking: 'thinking',
    ai_speaking: 'speaking',
    interrupted: 'user',
    error: 'error',
  };

  let screen = 'lobby';
  let startTime = null;
  let socket = null;
  let reconnectDelay = 1000;
  let lastProgress = { asked: 0, total: 0 };
  let lastMicResult = null;

  // The rings + mascot image are built once as a standalone "figure" node
  // and moved between the lobby slot and the live #avatar slot as the
  // screen changes, so exactly one Jerry image ever exists in the DOM.
  // #avatar and lobbyPortraitSlot are containers the figure moves between -
  // never the figure itself (an earlier version appended #avatar into
  // #avatar, a no-op DOM operation that left the image with nowhere to go).
  const avatarFigure = document.createElement('div');
  avatarFigure.className = 'avatar-figure';
  const avatarRingOuter = document.createElement('div');
  avatarRingOuter.className = 'ring ring-outer';
  const avatarRingMid = document.createElement('div');
  avatarRingMid.className = 'ring ring-mid';
  const avatarPortrait = document.createElement('div');
  avatarPortrait.className = 'portrait';
  const avatarImg = document.createElement('img');
  avatarImg.src = '/assets/jerry.png';
  avatarImg.alt = 'Jerry, your AI interviewer';
  avatarPortrait.appendChild(avatarImg);
  avatarFigure.append(avatarRingOuter, avatarRingMid, avatarPortrait);
  // Initial screen is 'lobby', so the figure belongs in the lobby slot from
  // the start.
  lobbyPortraitSlot.appendChild(avatarFigure);

  function showScreen(next) {
    if (screen === next) return;
    screen = next;
    lobby.hidden = next !== 'lobby';
    stage.hidden = next !== 'stage';
    done.hidden = next !== 'done';
    headerProgress.hidden = next === 'lobby';
    elapsedEl.hidden = next === 'lobby';
    controls.hidden = next !== 'stage';

    const home = next === 'lobby' ? lobbyPortraitSlot : avatar;
    if (avatarFigure.parentElement !== home) home.appendChild(avatarFigure);
  }

  function setState(state) {
    if (screen === 'lobby') showScreen('stage');
    avatar.dataset.state = state;
    statusLabel.textContent = STATE_LABELS[state] || state;
    statusPill.className = 'status-pill ' + (PILL_CLASS[state] || '');
    if (state !== 'idle' && startTime === null) startClock();
  }

  function startClock() {
    startTime = Date.now();
    setInterval(() => {
      const secs = Math.floor((Date.now() - startTime) / 1000);
      const m = String(Math.floor(secs / 60)).padStart(2, '0');
      const s = String(secs % 60).padStart(2, '0');
      elapsedEl.textContent = `${m}:${s}`;
    }, 1000);
  }

  function addBubble(role, text, cls = '') {
    if (!text) return;
    if (transcriptEmpty) transcriptEmpty.remove();
    const row = document.createElement('div');
    row.className = `bubble-row ${role}`;
    const bubble = document.createElement('div');
    bubble.className = `bubble ${cls}`.trim();
    if (role !== 'system') {
      const speaker = document.createElement('span');
      speaker.className = 'bubble-speaker';
      speaker.textContent = role === 'agent' ? 'Jerry' : 'You';
      bubble.appendChild(speaker);
    }
    const body = document.createElement('span');
    body.textContent = text;
    bubble.appendChild(body);
    row.appendChild(bubble);
    transcript.appendChild(row);

    const nearBottom =
      transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 140;
    if (nearBottom) transcript.scrollTop = transcript.scrollHeight;

    if (role === 'agent') lastLine.textContent = text;
  }

  function updateProgress(progress) {
    if (!progress || !progress.total) return;
    lastProgress = progress;
    const asked = Math.min(progress.asked, progress.total);
    progressLabel.textContent = `Question ${Math.max(asked, 1)} of ${progress.total}`;
    const pct = Math.min(100, (progress.asked / progress.total) * 100);
    progressFill.style.width = `${pct}%`;
  }

  // "Microphone Array (Intel® Smart..." -> "Intel Smart" - the raw device
  // string is real hardware naming and reads as a technical log line, not
  // something a candidate needs to parse.
  function humanizeDeviceName(name) {
    return name
      .replace(/\(.*?\)/g, '')
      .replace(/microphone|speakers?|array/gi, '')
      .replace(/\s+/g, ' ')
      .trim() || name;
  }

  function applyMicResult(msg, { dotEl, statusEl, subEl, containerEl }) {
    if (msg.ok) {
      dotEl.style.background = '';
      if (containerEl) containerEl.className = containerEl.className.replace(/\bmic-check\b/, 'mic-check ok').replace(/error/, '').trim();
      statusEl.textContent = 'Microphone connected';
      if (subEl) subEl.textContent = humanizeDeviceName(msg.mic);
    } else {
      if (containerEl) containerEl.className = containerEl.className.replace(/ok/, '').trim() + ' error';
      statusEl.textContent = 'Microphone issue detected';
      if (subEl) subEl.textContent = 'Check that a microphone is connected and try again.';
    }
  }

  function handleMicCheck(msg) {
    lastMicResult = msg;
    micCheck.hidden = false;
    applyMicResult(msg, {
      dotEl: micCheck.querySelector('.status-dot'),
      statusEl: micCheckLabel,
      subEl: micCheckSub,
      containerEl: micCheck,
    });
    if (msg.ok) {
      lobbyTitle.textContent = 'All set';
      lobbyCopy.textContent = 'The interview will begin shortly.';
    } else {
      lobbyTitle.textContent = 'Microphone issue detected';
      lobbyCopy.textContent = 'Fix the microphone issue below, then restart the interview.';
    }
  }

  function openMicModal() {
    micModalBackdrop.hidden = false;
    if (lastMicResult) {
      applyMicResult(lastMicResult, {
        dotEl: modalMicDot,
        statusEl: modalMicStatus,
        subEl: modalMicDevice,
      });
      modalMicDot.style.background = lastMicResult.ok ? 'var(--c-listening)' : 'var(--c-danger)';
    } else {
      modalMicDot.style.background = 'var(--c-inactive)';
      modalMicStatus.textContent = 'Checking…';
      modalMicDevice.textContent = '';
    }
  }

  function handleDone(msg) {
    updateProgress(msg.progress);
    const secs = startTime ? Math.floor((Date.now() - startTime) / 1000) : 0;
    const m = String(Math.floor(secs / 60)).padStart(2, '0');
    const s = String(secs % 60).padStart(2, '0');
    doneDuration.textContent = `${m}:${s}`;
    doneQuestions.textContent = `${lastProgress.asked} / ${lastProgress.total}`;
    showScreen('done');
  }

  function handleEvent(msg) {
    const { kind, data = {}, progress } = msg;
    updateProgress(progress);

    switch (kind) {
      case 'agent':
        addBubble('agent', data.text);
        if (data.interrupted) addBubble('system', 'candidate took the floor');
        break;
      case 'user':
        addBubble('user', data.text);
        break;
      case 'guardrail':
        addBubble('system', `guardrail: ${data.concern}`);
        break;
      case 'interrupted':
        addBubble('system', 'candidate took the floor');
        break;
      case 'latency':
        break; // internal timing detail, not shown to the candidate view
      default:
        break;
    }
  }

  function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    socket = new WebSocket(`${proto}://${location.host}/ws`);

    socket.addEventListener('open', () => {
      connWrap.className = 'conn live';
      connLabel.textContent = 'Connected';
      reconnectDelay = 1000;
    });

    socket.addEventListener('message', (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch {
        return;
      }
      if (msg.type === 'state') setState(msg.state);
      else if (msg.type === 'event') handleEvent(msg);
      else if (msg.type === 'mic_check') handleMicCheck(msg);
      else if (msg.type === 'done') handleDone(msg);
    });

    socket.addEventListener('close', () => {
      connWrap.className = 'conn down';
      connLabel.textContent = 'Reconnecting…';
      setTimeout(connect, reconnectDelay);
      reconnectDelay = Math.min(reconnectDelay * 1.5, 8000);
    });

    socket.addEventListener('error', () => socket.close());
  }

  // -- Test microphone modal ------------------------------------------
  testMicBtn.addEventListener('click', openMicModal);
  micModalClose.addEventListener('click', () => { micModalBackdrop.hidden = true; });
  micModalBackdrop.addEventListener('click', (e) => {
    if (e.target === micModalBackdrop) micModalBackdrop.hidden = true;
  });

  // -- End interview -----------------------------------------------------
  endBtn.addEventListener('click', () => { endModalBackdrop.hidden = false; });
  endCancel.addEventListener('click', () => { endModalBackdrop.hidden = true; });
  endModalBackdrop.addEventListener('click', (e) => {
    if (e.target === endModalBackdrop) endModalBackdrop.hidden = true;
  });
  endConfirm.addEventListener('click', async () => {
    endConfirm.disabled = true;
    endConfirm.textContent = 'Ending…';
    statusLabel.textContent = 'Ending the interview…';
    try {
      await fetch('/api/end-interview', { method: 'POST' });
    } catch {
      // The interview graph checks STOP_REQUESTED on its own cadence, so a
      // dropped response here does not mean the request failed to land.
    }
    endModalBackdrop.hidden = true;
  });

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    if (!micModalBackdrop.hidden) micModalBackdrop.hidden = true;
    if (!endModalBackdrop.hidden) endModalBackdrop.hidden = true;
  });

  connect();
})();
