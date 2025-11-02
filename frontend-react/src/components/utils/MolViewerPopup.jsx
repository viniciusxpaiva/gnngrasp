/* eslint-disable jsx-a11y/anchor-is-valid */
import React, { useEffect, useState } from "react";
import * as NGL from "ngl/dist/ngl";
import NGLViewer from "../visualization/NGLViewer";
import ResiduesTabs from "../items/ResiduesTabs";

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

export default function MolViewerPopup(props) {
  const [stagePopup, setStagePopup] = useState(null);
  const [tabIndex, setTabIndex] = useState(0);

  function changeColorBindSitesPopup(component, BindSites, color) {
    // Generate strings for each list inside bindSites
    const transformedArray = BindSites.map((item) => {
      const parts = item.split("-");
      return `${parts[1]}:${parts[2]}`;
    });

    //const bindSitesToShow = transformedArray.join(' or ');
    const bindSitesToShow = [transformedArray.join(" or ")];
    // Log the result strings
    bindSitesToShow.forEach((site, index) => {
      component.addRepresentation("ball+stick", {
        color: color,
        sele: site,
      });
    });
  }

  function colorAllSites(component) {
    if (props.predsToShow.includes("GRaSP"))
      changeColorBindSitesPopup(component, props.graspSites[0], bSiteColors[0]);
    if (props.predsToShow.includes("PUResNet"))
      changeColorBindSitesPopup(component, props.puresnetSites[0], bSiteColors[1]);
    if (props.predsToShow.includes("DeepPocket"))
      changeColorBindSitesPopup(component, props.deeppocketSites[0], bSiteColors[3]);
    if (props.predsToShow.includes("PointSite"))
      changeColorBindSitesPopup(component, props.pointsiteSites[0], bSiteColors[4]);
    if (props.predsToShow.includes("P2Rank"))
      changeColorBindSitesPopup(component, props.p2rankSites[0], "pink");
  }

  useEffect(() => {
    const newStage = new NGL.Stage("viewport-pop");
    newStage.removeAllComponents(); // Remove previous components
    newStage
      .loadFile(
        pdbFilesPath + "/pdbs/" + props.pdbFolder + "/AF-" + props.pdb + "-F1-model_v4.pdb"
      )
      .then((component) => {
        component.addRepresentation("cartoon", { color: "lightgrey" });
        component.autoView();
        colorAllSites(component);
        changeColorBindSitesPopup(
          component,
          props.upsetClickResidues,
          bSiteColors[5]
        );
      });
    newStage.setParameters({ backgroundColor: "white" });
    setStagePopup(newStage);
  }, []);

  return (
    <div className="row">
      <NGLViewer
        type={"popup"}
        stage={stagePopup}
        setStage={setStagePopup}
        pdb={props.pdb}
        pdbFolder={props.pdbFolder}
        bindSites={props.bindSites}
        graspSites={props.graspSites}
        puresnetSites={props.puresnetSites}
        deeppocketSites={props.deeppocketSites}
        pointsiteSites={props.pointsiteSites}
        p2rankSites={props.p2rankSites}
        predsToShow={props.predsToShow}
        upsetClickResidues={props.upsetClickResidues}
        puresnetButton={props.puresnetButton}
        graspButton={props.graspButton}
        pointsiteButton={props.pointsiteButton}
        p2rankButton={props.p2rankButton}
        deeppocketButton={props.deeppocketButton}
      />
      <ResiduesTabs
        type={"popup"}
        pdb={props.pdb}
        pred={props.pred}
        pdbFolder={props.pdbFolder}
        tabIndex={tabIndex}
        setTabIndex={setTabIndex}
        bindSites={props.bindSites}
        numPreds={props.numPreds}
        stage={stagePopup}
        upsetClickResidues={props.upsetClickResidues}
      />
    </div>
  );
}
