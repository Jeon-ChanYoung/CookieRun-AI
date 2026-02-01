let ws = null;
let noneInterval = null;
let slideInterval = null;
let isSliding = false;

let fpsCounter = 0;
let lastFpsTime = performance.now();

const ACTION_REPEAT_INTERVAL = 50; // milliseconds

// #################### WebSocket ####################

function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    ws = new WebSocket(`${protocol}//${host}/ws`);

    ws.onopen = () => {
        console.log("WebSocket connected");
        resetGame();
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateHUDAction(data.current_action);

        if (data.status === "error") {
            alert(data.message);
            return;
        }

        requestAnimationFrame(() => {
            document.getElementById('game-image').src = data.image;
            countFPS();
        });
    };

    ws.onclose = () => {
        console.log("Disconnected, reconnecting...");
        stopAllActions();
        setTimeout(connectWebSocket, 1000);
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
    };
}

function sendAction(action) {
    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'action', action }));
    }
}

// HUD update
function updateHUDAction(action) {
    const hudAction = document.getElementById('hud-action');
    hudAction.textContent = `${action}`;
}

// FPS counter
function countFPS() {
    fpsCounter++;
    const now = performance.now();

    if (now - lastFpsTime >= 500) {
        document.getElementById("hud-fps").textContent = fpsCounter * 2;
        fpsCounter = 0;
        lastFpsTime = now;
    }
}


// #################### None Action ####################

function startNoneAction() {
    if (noneInterval || isSliding) return;
    
    sendAction('none');
    noneInterval = setInterval(() => sendAction('none'), ACTION_REPEAT_INTERVAL);
}

function stopNoneAction() {
    if (noneInterval) {
        clearInterval(noneInterval);
        noneInterval = null;
    }
}

// #################### Jump Action ####################

function handleJump() {
    // 슬라이드 중이면 무시
    if (isSliding) return;
    
    stopNoneAction();
    sendAction('jump');
    
    // 점프 후 즉시 none 재시작
    startNoneAction();
    
    // 버튼 시각 피드백
    const btn = document.getElementById('btn-jump');
    btn.classList.add('active');
    setTimeout(() => btn.classList.remove('active'), 100);
}

// #################### Slide Action ####################

function startSlideAction() {
    if (isSliding) return;
    
    isSliding = true;
    stopNoneAction();
    
    sendAction('slide');
    slideInterval = setInterval(() => sendAction('slide'), ACTION_REPEAT_INTERVAL);
    
    document.getElementById('btn-slide').classList.add('active');
}

function stopSlideAction() {
    if (!isSliding) return;
    
    isSliding = false;
    
    if (slideInterval) {
        clearInterval(slideInterval);
        slideInterval = null;
    }
    
    document.getElementById('btn-slide').classList.remove('active');
    
    // 슬라이드 종료 후 none 재시작
    startNoneAction();
}

// #################### Reset & Stop All ####################

function stopAllActions() {
    stopNoneAction();
    stopSlideAction();
}

function resetGame() {
    stopAllActions();
    
    if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'reset' }));
    }

    const btn = document.getElementById('btn-reset');
    btn.classList.add('active');
    setTimeout(() => btn.classList.remove('active'), 100);
    
    // 리셋 후 none 시작
    startNoneAction();
}

// #################### Event Listeners ####################
// Jump button 
const jumpBtn = document.getElementById('btn-jump');

jumpBtn.addEventListener('mousedown', (e) => {
    e.preventDefault();
    handleJump();
});

jumpBtn.addEventListener('touchstart', (e) => {
    e.preventDefault();
    handleJump();
}, { passive: false });

// Slide button
const slideBtn = document.getElementById('btn-slide');

slideBtn.addEventListener('mousedown', (e) => {
    e.preventDefault();
    startSlideAction();
});

slideBtn.addEventListener('mouseup', (e) => {
    e.preventDefault();
    stopSlideAction();
});

slideBtn.addEventListener('mouseleave', (e) => {
    if (isSliding) stopSlideAction();
});

slideBtn.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startSlideAction();
}, { passive: false });

slideBtn.addEventListener('touchend', (e) => {
    e.preventDefault();
    stopSlideAction();
}, { passive: false });

slideBtn.addEventListener('touchcancel', (e) => {
    e.preventDefault();
    stopSlideAction();
}, { passive: false });

// Reset button
document.getElementById('btn-reset').addEventListener('click', (e) => {
    e.preventDefault();
    resetGame();
});

// #################### Keyboard Support ####################

const slideKeys = new Set(['s', 'S', 'ArrowDown']);
const jumpKeys = new Set(['w', 'W', 'ArrowUp', ' ']);
const resetKeys = new Set(['r', 'R']);

let keySliding = false;

document.addEventListener('keydown', (e) => {
    // Reset
    if (resetKeys.has(e.key)) {
        e.preventDefault();
        resetGame();
        return;
    }
    
    if (jumpKeys.has(e.key) && !e.repeat) {
        e.preventDefault();
        handleJump();
        return;
    }
    
    // Slide (hold)
    if (slideKeys.has(e.key) && !keySliding) {
        e.preventDefault();
        keySliding = true;
        startSlideAction();
    }
});

document.addEventListener('keyup', (e) => {
    if (slideKeys.has(e.key)) {
        keySliding = false;
        stopSlideAction();
    }
});

window.addEventListener('blur', () => {
    keySliding = false;
    stopSlideAction();
});

// #################### Init ####################

window.addEventListener('load', connectWebSocket);
