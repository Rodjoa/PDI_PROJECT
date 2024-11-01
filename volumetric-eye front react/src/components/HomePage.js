import React from 'react';

function HomePage({ setView }) {
  return (
    <div className="central-panel">
      <h2>Bienvenido a VolumetricEye</h2>
      <p>Seleccione una opción para comenzar.</p>
      <button onClick={() => setView('busqueda')}>Cargar Imagen/Videos</button>
      <button onClick={() => setView('previsualizacion')}>Capturar Imagen/Videos</button>
      <button onClick={() => setView('deteccion')}>Procesar</button>
    </div>
  );
}

export default HomePage;
