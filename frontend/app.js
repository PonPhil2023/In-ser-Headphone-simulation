console.log("app.js loaded");

const API_BASE = `${location.protocol}//${location.hostname}:8000/api`;

const statusEl = document.getElementById("status");
const runBtn = document.getElementById("runBtn");

function num(id){ return parseFloat(document.getElementById(id).value); }
function intv(id){ return parseInt(document.getElementById(id).value, 10); }
function setStatus(msg){ if(statusEl) statusEl.textContent = msg; console.log("[status]", msg); }

const canvas = document.getElementById("chart");
if(!canvas) throw new Error("Canvas #chart not found");
if(typeof Chart === "undefined") throw new Error("Chart.js not loaded");

let chart = new Chart(canvas.getContext("2d"), {
  type:"line",
  data:{ datasets:[{ label:"SPL (dB)", data:[], pointRadius:0, borderWidth:2 }] },
  options:{
    responsive:true, animation:false, parsing:false,
    interaction:{ mode:"index", intersect:false },
    scales:{
      x:{ type:"logarithmic", title:{ display:true, text:"Frequency (Hz)" } },
      y:{ title:{ display:true, text:"dB" } }
    }
  }
});

async function simulate(){
  setStatus("Running...");

  const payload = {
    f_min: num("fmin"),
    f_max: num("fmax"),
    n_points: intv("npoints"),
    bass_gain_db: num("bass"),
    treble_gain_db: num("treble"),
    resonance_hz: num("resHz"),
    resonance_q: num("resQ"),
    resonance_gain_db: num("resGain"),
  };

  try{
    const res = await fetch(`${API_BASE}/simulate`, {
      method:"POST",
      headers:{ "Content-Type":"application/json" },
      body: JSON.stringify(payload)
    });

    if(!res.ok){
      setStatus(`API error ${res.status}: ${await res.text()}`);
      return;
    }

    const data = await res.json();
    const points = data.freqs.map((f,i)=>({ x:f, y:data.spl_db[i] }));

    chart.data.datasets[0].data = points;
    chart.update();

    setStatus(`Done (${points.length} points)`);
  }catch(e){
    setStatus(`Fetch failed: ${e}`);
    console.error(e);
  }
}

runBtn.addEventListener("click", simulate);
simulate();