import React, { useState, useEffect } from "react";
import { useNavigate } from 'react-router-dom';
import { useParams } from "react-router-dom";
import BaseLayout from "../components/layout/base";
import "reactjs-popup/dist/index.css";
import ResultsPageTabs from "../components/items/ResultsPageTabs";
import Backdrop from '@mui/material/Backdrop';
import CircularProgress from '@mui/material/CircularProgress';
import Tooltip, { tooltipClasses } from '@mui/material/Tooltip';
import { styled } from '@mui/system';

const predictors = [
  "GRaSP",
  "PUResNet",
  "DeepPocket",
  "PointSite",
  "P2Rank",
];

const Results = () => {
  const { inputString } = useParams();

  const [graspSites, setGraspSites] = useState([]);
  const [pdbFolder, setPdbFolder] = useState("");
  const [proteinFullName, setProteinFullName] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    // Fetch the processed string from the Flask backend
    const fetchProcessedString = async () => {
      try {
        const response = await fetch("/process", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ inputString }),
        });

        

        const data = await response.json();
        console.log(data)
        if (data.prot_folder.length === 0){
          navigate(`/notfound`);
        }
        setGraspSites(data.grasp);
        setPdbFolder("input");
        console.log(pdbFolder)
      } catch (error) {
        console.error("Error:", error);
      }
    };

    fetchProcessedString();
  }, []);

  return (
    <>
      <BaseLayout>
        <div
          className="container-fluid bg-light-dark text-white mt-0 py-4"
          id="help-submit"
        >
          <div className="row justify-content-center">
            <div class="col-md-12 text-center">
              {pdbFolder ? (
                <h6 className="display-6 text-light">
                  Predicted binding sites for input protein
                </h6>) : (
                <h6 className="display-6 text-light">
                  Searching results...
                </h6>)}
            </div>
          </div>
        </div>

        <div class="container-lg">
          {pdbFolder ? (
            <ResultsPageTabs
              predictors={predictors}
              pdb={inputString}
              pdbFolder={pdbFolder}
              graspSites={graspSites}
              proteinFullName={proteinFullName}
            />
          ) : (
            <div className="row mt-4">
              <Backdrop
                sx={{
                  color: '#fff',
                  zIndex: (theme) => theme.zIndex.drawer + 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
                open={true}
              >
                <div className="mb-4">
                  Please wait. Loading data...
                </div>
                <CircularProgress color="inherit" />
              </Backdrop>
            </div>
          )}
        </div>
      </BaseLayout>
    </>
  );
};
export default Results;
