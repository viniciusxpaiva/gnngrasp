import React from 'react';
import './styles/App.css';
import './styles/Base.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/home';
import Results from './pages/results';
import Contact from './pages/contact';
import Help from './pages/help';
import NotFound from './pages/notFound';
import DataTable from './pages/dataTable';
import ZeroPredict from './pages/zeroPredict';
import ReactGA from 'react-ga';
import usePageTracking from './components/items/PageTracking';

// Initialize Google Analytics
ReactGA.initialize('G-V8KQDT1TXL');

function App() {
  return (
    <Router>
      <PageTrackingWrapper>
        <Routes>
          <Route path='/' element={<Home />} />
          <Route path='/results/:inputString' element={<Results />} />
          <Route path='/contact' element={<Contact />} />
          <Route path='/help' element={<Help />} />
          <Route path='/notfound' element={<NotFound />} />
          <Route path='/datatable' element={<DataTable />} />
          <Route path='/nopredictions' element={<ZeroPredict />} />
        </Routes>
      </PageTrackingWrapper>
    </Router>
  );
}

// Create a wrapper component for page tracking
const PageTrackingWrapper = ({ children }) => {
  usePageTracking();
  return <>{children}</>;
};

export default App;
