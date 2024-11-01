import React, { useState } from 'react';
import BusquedaArchivos from './BusquedaArchivos';
import Previsualizacion from './Previsualizacion';
import PantallaResultados from './PantallaResultados';
import ProcesoDeteccion from './ProcesoDeteccion';
import HomePage from './HomePage';

function VolumetricEye() {
  const [view, setView] = useState('home');

  const renderView = () => {
    switch(view) {
      case 'busqueda':
        return <BusquedaArchivos setView={setView} />;
      case 'previsualizacion':
        return <Previsualizacion setView={setView} />;
      case 'resultados':
        return <PantallaResultados setView={setView} />;
      case 'deteccion':
        return <ProcesoDeteccion setView={setView} />;
      default:
        return <HomePage setView={setView} />;
    }
  };

  return (
    <div className="VolumetricEye">
      <header>
        <h1>VolumetricEye</h1>
      </header>
      {renderView()}
    </div>
  );
}

export default VolumetricEye;
