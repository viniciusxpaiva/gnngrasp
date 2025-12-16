import React, { useEffect, useState } from "react";
import * as NGL from "ngl/dist/ngl";
import Stack from "@mui/material/Stack";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import NGLViewer from "../visualization/NGLViewer";
import ResiduesTabs from "../items/ResiduesTabs";
import Card from "@mui/material/Card";
import Typography from "@mui/material/Typography";

//const pdbFilesPath = "https://benderdb.ufv.br/benderdb-data"
const pdbFilesPath = ""

const bSiteColors = [
  "#167288",
  "#a89a49",
  "#b45248",
  "#3cb464",
  "#643c6a",
  "#8cdaec",
  "#d48c84",
  "#d6cfa2",
  "#9bddb1",
  "#836394",
];

export default function MolViewerPredictors(props) {
  const [tabIndex, setTabIndex] = useState(0);
  const [stagePredictors, setStagePredictors] = useState("");

  function generateBindSiteStringPredictors(cluster) {
    const residues = Array.isArray(cluster) ? cluster : cluster?.residues || [];
    const stringArray = residues
      .map((item) => `${item[2]}:${item[0]}`)
      .join(" or ");
    return stringArray;
  }

  function changeColorBindSitesPredictors(component, BindSites) {
    // Generate strings for each list inside bindSites
    const bindSitesToShow = BindSites.map(generateBindSiteStringPredictors);
    bindSitesToShow.forEach((site, index) => {
      component.addRepresentation("ball+stick", {
        color: bSiteColors[index % bSiteColors.length],
        sele: site,
      });
    });
  }

  useEffect(() => {
    if (!props.pdb || !props.pdbFolder) return;

    const newStage = new NGL.Stage("viewport");
    newStage.removeAllComponents(); // Remove previous components
    newStage
      .loadFile(
        pdbFilesPath + "/pdbs/output_" + props.pdbFolder + "/" + props.pdb + ".pdb"
      )
      .then((component) => {
        component.addRepresentation("cartoon", { color: "lightgrey" });
        component.autoView();
        changeColorBindSitesPredictors(component, props.bindSites);
      })
      .catch((err) => console.error("Error loading PDB in NGL:", err));
    newStage.setParameters({ backgroundColor: "white" });
    setStagePredictors(newStage);

    return () => {
      newStage.dispose();
    };
  }, [props.pdb, props.pdbFolder, props.bindSites]);

  return (
    <>
      {props.bindSites.length > 0 ? (
        <div className="row">
          <NGLViewer
            type={"predictors"}
            pdb={props.pdb}
            pdbFolder={props.pdbFolder}
            pred={props.pred}
            numPreds={props.numPreds}
            consensusData={props.consensusData}
            stage={stagePredictors}
            setStage={setStagePredictors}
            tabIndex={tabIndex}
            setTabIndex={setTabIndex}
            bindSites={props.bindSites}
            bSiteColors={bSiteColors}
          />

          <ResiduesTabs
            type={"predictors"}
            pdb={props.pdb}
            pred={props.pred}
            pdbFolder={props.pdbFolder}
            tabIndex={tabIndex}
            setTabIndex={setTabIndex}
            bindSites={props.bindSites}
            numPreds={props.numPreds}
            stage={stagePredictors}
          />
        </div>
      ) : (
        <div className="row">
          <Stack sx={{ width: "100%" }} spacing={2}>
            <Alert severity="info">
              <AlertTitle>
                <strong>Info</strong>
              </AlertTitle>
              {props.pred} did not find any binding site for protein {props.pdb}
            </Alert>
          </Stack>
        </div>
      )}
    </>
  );
}
