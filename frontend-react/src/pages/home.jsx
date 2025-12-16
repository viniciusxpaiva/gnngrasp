import React, { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import BaseLayout from "../components/layout/base";
import Paper from "@mui/material/Paper";
import InputBase from "@mui/material/InputBase";
import IconButton from "@mui/material/IconButton";
import SearchIcon from "@mui/icons-material/Search";
import Button from "@mui/material/Button";
import Typography from "@mui/material/Typography";
import Chip from "@mui/material/Chip";

const Home = () => {
    const [searchString, setSearchString] = useState("");
    const [uploadedFile, setUploadedFile] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const fileInputRef = useRef(null);
    const navigate = useNavigate();

    const submitToProcess = async (code, file) => {
        if (!code && !file) return;
        setIsSubmitting(true);
        try {
            let response;
            if (file) {
                const formData = new FormData();
                formData.append("pdbFile", file);
                formData.append("pdbCode", code);
                response = await fetch("/process", { method: "POST", body: formData });
            } else {
                response = await fetch("/process", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ pdbCode: code }),
                });
            }

            if (!response.ok) {
                navigate(`/notfound`);
                return;
            }

            const data = await response.json();
            if (data.error || !data.job_id) {
                navigate(`/notfound`);
                return;
            }

            navigate(`/results/${data.job_id}`, {
                replace: true,
                state: {
                    pdbCode: data.pdb_code,
                    jobData: data,
                    uploadedFileName: file?.name || null,
                },
            });
        } catch (err) {
            console.error(err);
            navigate(`/notfound`);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handlePredictSubmit = (e) => {
        e.preventDefault();
        const trimmed = searchString.trim().toLowerCase();
        if (!trimmed && !uploadedFile) return;
        submitToProcess(trimmed, uploadedFile);
    };

    const handleFileUpload = (event) => {
        const file = event.target.files && event.target.files[0];
        if (!file) return;
        const cleanName = file.name.replace(/\.pdb$/i, "");
        setUploadedFile(file);
        setSearchString(cleanName.trim().toLowerCase());
    };

    const handleClearFile = () => {
        setUploadedFile(null);
        setSearchString("");
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    const handleExampleClick = (pdb) => {
        const code = pdb.trim().toLowerCase();
        submitToProcess(code, null);
    };

    const hasInput = Boolean(searchString || uploadedFile);

    return (
        <BaseLayout>
            <div className="jumbotron bg-light-dark">
                <div className="container">
                    <div className="row mt-6">
                        <div className="col-md-12 text-center">
                            <h1 className="display-4 text-light mt-5">
                                <strong>GNN-GRaSP</strong>
                            </h1>
                            <p
                                className="display-7 text-light mt-3"
                                style={{ fontSize: "22px" }}
                            >
                                a protein binding site predictor using hierarchical graph neural networks
                            </p>

                            <div className="container p-0 mb-5 mt-5 justify-content-center">
                                <div className="row justify-content-center">
                                    <div className="col-12 col-md-10 col-lg-8">
                                        {/* ----- PDB code OU upload de PDB no mesmo Paper ----- */}
                                        <Paper
                                            component="form"
                                            sx={{
                                                p: "2px 4px",
                                                display: "flex",
                                                alignItems: "center",
                                                width: "100%",
                                                marginBottom: "10px",
                                                backgroundColor: "white",
                                                flexWrap: "wrap",
                                                gap: 1,
                                            }}
                                            onSubmit={handlePredictSubmit}
                                        >
                                            <IconButton
                                                type="button"
                                                sx={{ p: "10px" }}
                                                aria-label="search"
                                                disabled
                                            >
                                                <SearchIcon />
                                            </IconButton>

                                            <InputBase
                                                value={uploadedFile ? "" : searchString}
                                                onChange={(e) => {
                                                    setUploadedFile(null);
                                                    if (fileInputRef.current) {
                                                        fileInputRef.current.value = "";
                                                    }
                                                    setSearchString(e.target.value.toLowerCase());
                                                }}
                                                sx={{ ml: 1, flex: 1 }}
                                                placeholder={
                                                    uploadedFile
                                                        ? `Uploaded file: ${uploadedFile.name}`
                                                        : "Enter PDB code or upload a PDB file"
                                                }
                                                inputProps={{
                                                    "aria-label": "search PDB code",
                                                    readOnly: !!uploadedFile,
                                                }}
                                            />

                                            {uploadedFile && (
                                                <Chip
                                                    label={uploadedFile.name}
                                                    onDelete={handleClearFile}
                                                    size="small"
                                                    sx={{ maxWidth: "45%" }}
                                                />
                                            )}

                                            <input
                                                type="file"
                                                accept=".pdb"
                                                ref={fileInputRef}
                                                style={{ display: "none" }}
                                                onChange={handleFileUpload}
                                            />
                                            <Button
                                                variant="outlined"
                                                type="button"
                                                onClick={() => fileInputRef.current?.click()}
                                                disabled={isSubmitting}
                                            >
                                                Upload PDB
                                            </Button>
                                            <Button
                                                variant="contained"
                                                type="submit"
                                                disabled={!hasInput || isSubmitting}
                                            >
                                                {isSubmitting ? "Processing..." : "Predict"}
                                            </Button>
                                        </Paper>


                                        {/* EXEMPLOS (PDB CODES) */}
                                        <Paper
                                            sx={{
                                                p: "2px 4px",
                                                display: "flex",
                                                alignItems: "center",
                                                width: "100%",
                                                backgroundColor: "inherit",
                                                mb: 3,
                                            }}
                                            elevation={0}
                                        >
                                            <Typography
                                                variant="body2"
                                                sx={{ color: "white", mr: 1 }}
                                            >
                                                Examples:
                                            </Typography>
                                            <Button
                                                variant="outlined"
                                                onClick={() => handleExampleClick("1a8t")}
                                                sx={{ color: "white", borderColor: "white", mr: 2 }}
                                            >
                                                1a8t
                                            </Button>
                                            <Button
                                                variant="outlined"
                                                onClick={() => handleExampleClick("2aai")}
                                                sx={{ color: "white", borderColor: "white", mr: 2 }}
                                            >
                                                2aai
                                            </Button>
                                            <Button
                                                variant="outlined"
                                                onClick={() => handleExampleClick("3nos")}
                                                sx={{ color: "white", borderColor: "white" }}
                                            >
                                                3nos
                                            </Button>
                                        </Paper>
                                    </div>
                                </div>
                            </div>



                        </div>
                    </div>
                </div>

            </div>
            <div class="container">
                {/* SECTION 1 — About GNN-GRaSP */}
                <div class="row mt-4">
                    <div
                        class="col-md-6"
                        style={{
                            display: "flex",
                            flexDirection: "column",
                            justifyContent: "center",
                        }}
                    >
                        <p>
                            <h2>Protein Binding Site Prediction</h2>
                        </p>
                        <p>
                            <strong>GNN-GRaSP</strong> is a graph-based deep learning framework
                            designed to identify ligand-binding residues in protein structures.
                        </p>
                        <p>
                            The method models each protein as a graph, where residues are nodes and
                            non-covalent interactions define the edges. It uses <b>Graph Neural
                                Networks (GNNs)</b> combined with a hierarchical learning strategy to
                            enhance detection of binding regions, especially under severe class
                            imbalance.
                        </p>
                        <p>
                            The predictor leverages the <b>CLARA framework</b>, which integrates
                            subgraph decomposition and coarse-to-fine learning to focus the model on
                            structurally relevant regions of the protein.
                        </p>
                    </div>
                    <div class="col-md-6">
                        <div class="bordered">
                            <img
                                src="img/ngl_home2.png"
                                className="img-fluid"
                                style={{
                                    maxWidth: "100%",
                                    maxHeight: "320px",
                                    width: "auto",
                                    height: "auto",
                                }}
                                alt="Protein structure viewer"
                            />
                        </div>
                    </div>
                </div>

                <hr />

                {/* SECTION 2 — Framework Overview */}
                <div class="row mb">
                    <div class="col-12 text-center">
                        <h2 style={{ marginBottom: "20px" }}>Hierarchical Graph Framework</h2>
                        <div class="bordered mb-3" style={{ display: "inline-block" }}>
                            <img
                                src="img/hierarchical.png"
                                class="img-fluid"
                                alt="Framework design"
                                style={{ display: "block", margin: "0 auto" }}
                            />
                        </div>
                        <p>
                            GNN-GRaSP employs a <b>two-stage hierarchical model</b> based on CLARA:
                            (i) a <b>subgraph-level classifier</b> identifies regions likely to
                            contain binding residues, and (ii) a <b>node-level classifier</b> refines
                            predictions within those subgraphs.
                        </p>
                        <p>
                            Subgraphs can be generated through <b>randomized coverage</b> or guided
                            by <b>solvent-accessibility heuristics</b>, prioritizing surface regions
                            where ligand interactions are more likely.
                        </p>
                        <p>
                            This hierarchical design improves prediction precision, reduces noise
                            from non-binding regions, and provides interpretable residue-level
                            outputs across different protein structures.
                        </p>
                    </div>
                </div>

                <hr />

                {/* SECTION 3 — Results and Visualization */}
                <div class="row mt-1">
                    <div class="col-md-6">
                        <div class="bordered">
                            <img
                                src="img/visu.png"
                                className="img-fluid"
                                style={{
                                    maxWidth: "100%",
                                    maxHeight: "500px",
                                    width: "auto",
                                    height: "auto",
                                }}
                                alt="Visualization of predictions"
                            />
                        </div>
                    </div>
                    <div
                        class="col-md-6"
                        style={{
                            display: "flex",
                            flexDirection: "column",
                        }}
                    >
                        <br />
                        <br />
                        <h2 style={{ marginBottom: "25px" }}>Results and Visualization</h2>
                        <p>
                            GNN-GRaSP achieved <b>state-of-the-art performance</b> on the COACH100
                            benchmark, surpassing existing predictors such as PUResNet and GRaSP,
                            with an MCC of <b>0.674</b> and an F1-score of <b>68.6%</b>.
                        </p>
                        <p>
                            The integrated molecular viewer allows users to explore predicted
                            binding residues directly on the protein structure, highlighting
                            high-confidence binding regions in warm colors and non-binding areas in
                            cooler tones.
                        </p>
                        <p>
                            Interactive plots, such as the UpSet view, provide insight into how
                            predictions overlap across subgraphs and classifiers, supporting both
                            visual interpretation and comparative analysis.
                        </p>
                    </div>
                </div>
            </div>
        </BaseLayout>
    );
};

export default Home;
