import React from 'react';

function ProcesoDeteccion({ setView }) {
  return (
    <div className="central-panel">
      <h2>Proceso de Detección y Segmentación</h2>
      <p>Configuración y ejecución del procesamiento.</p>
      <button onClick={() => setView('resultados')}>Ver Resultados</button>
      <button onClick={() => setView('home')}>Regresar</button>
    </div>
  );
}

export default ProcesoDeteccion;
