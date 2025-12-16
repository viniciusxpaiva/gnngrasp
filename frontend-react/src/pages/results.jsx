import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from 'react-router-dom';
import { useParams } from "react-router-dom";
import BaseLayout from "../components/layout/base";
import "reactjs-popup/dist/index.css";
import ResultsPageTabs from "../components/items/ResultsPageTabs";
import Backdrop from '@mui/material/Backdrop';
import CircularProgress from '@mui/material/CircularProgress';
import Tooltip, { tooltipClasses } from '@mui/material/Tooltip';
import { styled } from '@mui/system';
import Button from "@mui/material/Button";

const predictors = [
  "GRaSP",
  "PUResNet",
  "DeepPocket",
  "PointSite",
  "P2Rank",
];

const Results = () => {
  const { jobId } = useParams();
  const location = useLocation();
  const uploadedFileName = location.state?.uploadedFileName || "";
  const initialPdbCode = (location.state?.pdbCode || "").trim().toLowerCase();
  const prefetchedJob = location.state?.jobData;

  const [graspSites, setGraspSites] = useState([]);
  const [pdbFolder, setPdbFolder] = useState("");
  const [proteinFullName, setProteinFullName] = useState("");
  const [pdbCode, setPdbCode] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [copyMsg, setCopyMsg] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    // Fetch the processed string from the Flask backend
    const fetchProcessedString = async () => {
      try {
        setGraspSites([]);
        setPdbFolder("");
        setProteinFullName("");
        setPdbCode("");
        setIsRunning(false);
        setErrorMsg("");

        if (!jobId) {
          navigate(`/notfound`);
          return;
        }

        // Se já temos dados prefetchados (veio da Home), popula e evita /process extra
        if (prefetchedJob && prefetchedJob.job_id === jobId) {
          if (prefetchedJob.status === "RUNNING") {
            setIsRunning(true);
            setPdbFolder(prefetchedJob.job_id);
            setPdbCode((prefetchedJob.pdb_code || "").toUpperCase());
            // segue para consultar o backend para ver se já finalizou
          } else {
            setGraspSites(prefetchedJob.grasp);
            setPdbFolder(prefetchedJob.job_id);
            setProteinFullName(prefetchedJob.prot_full_name || "");
            setPdbCode((prefetchedJob.pdb_code || "").toUpperCase());
            return;
          }
        }

        const response = await fetch("/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ jobId }),
        });

        const data = await response.json();
        if (!response.ok) {
          if (data.status === "FAILED" || data.error === "JOB_FAILED") {
            setErrorMsg("The job failed during execution. Please try again with another PDB.");
            return;
          }
          if (data.error === "JOB_NOT_FOUND") {
            setErrorMsg("Job not found. Please verify the link or run a new prediction.");
            return;
          }
          navigate(`/notfound`);
          return;
        }
        console.log(data);

        if (data.error) {
          if (data.status === "FAILED" || data.error === "JOB_FAILED") {
            setErrorMsg("The job failed during execution. Please try again with another PDB.");
            return;
          }
          if (data.error === "JOB_NOT_FOUND") {
            setErrorMsg("Job not found. Please verify the link or run a new prediction.");
            return;
          }
          navigate(`/notfound`);
          return;
        }
        if (data.status === "RUNNING") {
          setIsRunning(true);
          setPdbFolder(data.job_id);
          setPdbCode((data.pdb_code || "").toUpperCase());
          return;
        }
        const noSites =
          !data.grasp ||
          data.grasp.length === 0 ||
          data.grasp.every((site) => {
            const residues = Array.isArray(site)
              ? site
              : site?.residues || [];
            return residues.length === 0;
          });

        if (!data.prot_folder || data.prot_folder.length === 0 || noSites){
          navigate(`/notfound`);
          return;
        }
        setGraspSites(data.grasp);
        setPdbFolder(data.job_id);
        setProteinFullName(data.prot_full_name || "");
        setPdbCode((data.pdb_code || "").toUpperCase());

      } catch (error) {
        console.error("Error:", error);
      }
    };

    fetchProcessedString();
  }, [jobId, navigate, prefetchedJob, initialPdbCode]);

  const resolvedPdbCode = (pdbCode || initialPdbCode || "").toLowerCase();
  const displayName = uploadedFileName || resolvedPdbCode.toUpperCase();
  const jobLink =
    (typeof window !== "undefined" && jobId)
      ? `${window.location.origin}/results/${jobId}`
      : "";

  const handleCopyLink = async () => {
    if (!jobLink) return;
    try {
      await navigator.clipboard.writeText(jobLink);
      setCopyMsg("Link copied!");
    } catch (err) {
      console.error("Clipboard error:", err);
      setCopyMsg("Unable to copy, please copy manually.");
    }
    setTimeout(() => setCopyMsg(""), 2000);
  };

  // Polling while job is running
  useEffect(() => {
    if (!jobId || !isRunning) return;

    let isCancelled = false;
    const poll = async () => {
      try {
        const res = await fetch("/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ jobId }),
        });
        const data = await res.json();

        if (!res.ok) {
          if (data.status === "FAILED" || data.error === "JOB_FAILED") {
            if (!isCancelled) {
              setErrorMsg("The job failed during execution. Please try again with another PDB.");
              setIsRunning(false);
            }
          }
          return;
        }

        if (data.status === "RUNNING") {
              if (!isCancelled) {
                setIsRunning(true);
              }
              return;
            }

            if (data.status === "DONE" && !isCancelled) {
              const noSites =
                !data.grasp ||
                data.grasp.length === 0 ||
                data.grasp.every((site) => {
                  const residues = Array.isArray(site)
                    ? site
                    : site?.residues || [];
                  return residues.length === 0;
                });

              if (!data.prot_folder || noSites) {
                navigate(`/notfound`);
                return;
              }
              setGraspSites(data.grasp);
              setPdbFolder(data.job_id);
          setProteinFullName(data.prot_full_name || "");
          setPdbCode((data.pdb_code || "").toUpperCase());
          setIsRunning(false);
        }
      } catch (err) {
        console.error(err);
      }
    };

    poll();
    const intervalId = setInterval(poll, 5000);
    return () => {
      isCancelled = true
      clearInterval(intervalId);
    };
  }, [jobId, isRunning, navigate]);

  return (
    <>
      <BaseLayout>
        <div
          className="container-fluid bg-light-dark text-white mt-0 py-4"
          id="help-submit"
        >
          <div className="row justify-content-center">
            <div class="col-md-12 text-center">
              {errorMsg ? (
                <h6 className="display-6 text-light">
                  {errorMsg}
                </h6>
              ) : isRunning ? (
                <h6 className="display-6 text-light">
                  GNN-GRaSP is running for {displayName}
                </h6>
              ) : pdbFolder ? (
                <h6 className="display-6 text-light">
                  Predicted binding sites for {uploadedFileName ? "uploaded file" : "protein"}: {displayName}
                </h6>) : (
                <h6 className="display-6 text-light">
                  Searching results...
                </h6>)}
            </div>
          </div>
        </div>

        <div class="container-lg">
          {errorMsg ? (
            <div className="row mt-4 text-center text-light">
              <p>{errorMsg}</p>
            </div>
          ) : isRunning ? (
            <div className="row mt-4">
              <Backdrop
                sx={{
                  color: '#fff',
                  zIndex: (theme) => theme.zIndex.drawer + 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 2,
                }}
                open={true}
              >
                <div style={{ textAlign: "center", lineHeight: 1.4 }}>
                  <div>GNN-GRaSP is running. Results will show automatically once finished.</div>
                  <div style={{marginTop: "4px" }}>
                    Predictions may take up to 1 minute.
                  </div>
                </div>
                <CircularProgress color="inherit" />
                {jobLink && (
                  <div style={{ textAlign: "center", marginTop: "10px" }}>
                    <div style={{ marginBottom: "6px" }}>Job link (share or open later):</div>
                    <div style={{ wordBreak: "break-all", fontSize: "0.9rem" }}>{jobLink}</div>
                    <Button
                      variant="contained"
                      color="primary"
                      size="small"
                      onClick={handleCopyLink}
                      sx={{ mt: 1 }}
                    >
                      Copy link
                    </Button>
                    {copyMsg && (
                      <div style={{ marginTop: "4px", fontSize: "0.85rem" }}>{copyMsg}</div>
                    )}
                  </div>
                )}
              </Backdrop>
            </div>
          ) : pdbFolder ? (
            <ResultsPageTabs
              predictors={predictors}
              pdb={resolvedPdbCode}
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
                  Please wait. Running GNN-GRaSP...
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
