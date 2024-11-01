import React from 'react';

function PantallaResultados({ setView }) {
  return (
    <div className="central-panel">
      <h2>Resultados</h2>
      <p>Aquí se muestran los resultados del procesamiento.</p>
      <button onClick={() => alert("Resultados guardados")}>Guardar Resultados</button>
      <button onClick={() => setView('home')}>Regresar</button>
    </div>
  );
}

export default PantallaResultados;
