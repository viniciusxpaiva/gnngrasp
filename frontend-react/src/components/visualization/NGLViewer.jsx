import React, { useState, useEffect } from "react";
import Stack from "@mui/material/Stack";
import IconButton from "@mui/material/IconButton";
import RestartAltIcon from "@mui/icons-material/RestartAlt";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import Select from "@mui/material/Select";
import Button from "@mui/material/Button";
import DownloadingIcon from "@mui/icons-material/Downloading";
import MouseHelpPopup from "../items/MouseHelpPopup";
import PropTypes from "prop-types";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import FormHelperText from "@mui/material/FormHelperText";
import Card from "@mui/material/Card";
import Divider from "@mui/material/Divider";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import SettingsIcon from "@mui/icons-material/Settings";
import CloseIcon from "@mui/icons-material/Close";
import Tooltip, { tooltipClasses } from '@mui/material/Tooltip';
import { styled } from '@mui/system';

//const pymolDLUrl = "https://benderdb.ufv.br/benderdb-data/zip_pymol/"
const pymolDLUrl = process.env.PUBLIC_URL + "/zip_pymol/"

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

const NoMaxWidthTooltip = styled(({ className, ...props }) => (
  <Tooltip {...props} classes={{ popper: className }} arrow />
))({
  [`& .${tooltipClasses.tooltip}`]: {
    maxWidth: 'none',
  },
});

function ColorfulText({ color, hoverColor, children }) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <span
      style={{ color: isHovered ? hoverColor : color }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {children}
    </span>
  );
}

export default function NGLViewer(props) {
  const [protReprButton, setProtReprButton] = useState("surface");
  const [siteReprButton, setSiteProtReprButton] = useState("licorice");
  const [bgroundColor, setBGroundColor] = useState("white");
  const [open, setOpen] = useState(false);

  function setViewerTabs() {
    if (props.type === "summary") {
      return <ViewerTabsSummary />;
    } else if (props.type === "predictors") {
      return <ViewerTabsPredictors />;
    } else if (props.type === "popup") {
      return <ViewerTabsPopup />;
    } else {
      return null; // Or handle other cases
    }
  }

  function changeColorBindSites(component, BindSites, repr) {
    if (props.type === "summary") {
      changeColorBindSitesSummary(component, BindSites, repr);
    } else if (props.type === "predictors") {
      changeColorBindSitesPredictors(component, BindSites);
    }
  }

  function resetNGLViewer(stage, tabIndex) {
    if (props.type === "summary") {
      resetNGLViewerSummary(stage, tabIndex);
    } else if (props.type === "predictors") {
      resetNGLViewerPredictors(stage, tabIndex);
    } else if (props.type === "popup") {
      resetNGLViewerPopup(stage);
    }
  }

  function handleDownloadPymol(protName) {
    if (props.type === "summary") {
      handleDownloadPymolSummary(protName);
    } else if (props.type === "predictors") {
      handleDownloadPymolPredictors(protName);
    } else if (props.type === "popup") {
    }
  }

  function generateBindSiteStringSummary(bindSiteList) {
    const stringArray = bindSiteList
      .map((item) => `${item[2]}:${item[0]}`)
      .join(" or ");
    return stringArray;
  }

  function generateBindSiteStringPredictors(bindSiteList) {
    const stringArray = bindSiteList
      .map((item) => `${item[2]}:${item[0]}`)
      .join(" or ");
    return stringArray;
  }

  function changeColorBindSitesSummary(component, BindSites, repr) {
    // Generate strings for each list inside bindSites
    if (BindSites === null) {
      return;
    }
    const bindSitesToShow = generateBindSiteStringSummary(BindSites);
    component.addRepresentation("ball+stick", {
      color: "#b45248",
      sele: bindSitesToShow,
    });
  }

  function changeColorBindSitesPredictors(component, BindSites) {
    // Generate strings for each list inside bindSites
    const bindSitesToShow = BindSites.map(generateBindSiteStringPredictors);
    bindSitesToShow.forEach((site, index) => {
      component.addRepresentation("ball+stick", {
        color: props.bSiteColors[index % props.bSiteColors.length],
        sele: site,
      });
    });
  }

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

  function colorAllSitesPopup(component) {
    if (props.predsToShow.includes("GRaSP") && props.graspButton === "selected")
      changeColorBindSitesPopup(component, props.graspSites[0], bSiteColors[0]);
    if (
      props.predsToShow.includes("PUResNet") &&
      props.puresnetButton === "selected"
    )
      changeColorBindSitesPopup(
        component,
        props.puresnetSites[0],
        bSiteColors[1]
      );
    if (
      props.predsToShow.includes("DeepPocket") &&
      props.deeppocketButton === "selected"
    )
      changeColorBindSitesPopup(
        component,
        props.deeppocketSites[0],
        bSiteColors[3]
      );
    if (
      props.predsToShow.includes("PointSite") &&
      props.pointsiteButton === "selected"
    )
      changeColorBindSitesPopup(
        component,
        props.pointsiteSites[0],
        bSiteColors[4]
      );
    if (
      props.predsToShow.includes("P2Rank") &&
      props.p2rankButton === "selected"
    )
      changeColorBindSitesPopup(component, props.p2rankSites[0], "pink");
  }

  function handleDownloadPymolSummary(protName) {
    const fileUrl = pymolDLUrl + props.pdbFolder + "/" + protName + "_pymol_session.zip";
    const link = document.createElement("a");
    link.href = fileUrl;
    link.download = protName + "_pymol_session.zip";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function handleDownloadPymolPredictors(protName) {
    const fileUrl = pymolDLUrl + props.pdbFolder + "/" + protName + "_" + props.pred + "_sites_pymol_session.zip";
    console.log(fileUrl)
    const link = document.createElement("a");
    link.href = fileUrl;
    link.download = protName + "_" + props.pred + "_sites_pymol_session.zip";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function resetNGLViewerSummary(stage, tabIndex) {
    resetParameters();
    stage.removeAllComponents();

    if (tabIndex === 0) {
      stage
        .loadFile(
          pdbFilesPath + "/pdbs/" + props.pdbFolder + "/AF-" + props.pdb + "-F1-model_v4.pdb"
        )
        .then((component) => {
          component.addRepresentation("cartoon", {
            colorScheme: "bfactor",
            colorScale: "RdYlBu", // Defines a color scale from red to blue
            colorReverse: true, // Reverses the color scale to use blue for low bfactor values and red for high bfactor values
          });
          component.autoView();
          //changeColorBindSites(component, props.upsetClickResidues)
        });
    } else {
      const filteredData = props.consensusData.filter(
        (p) =>
          p[3] >=
          (props.numPreds * props.maxConsensusPercent - tabIndex + 2) /
          props.numPreds
      );
      stage
        .loadFile(
          pdbFilesPath + "/pdbs/" + props.pdbFolder + "/AF-" + props.pdb + "-F1-model_v4.pdb"
        )
        .then((component) => {
          component.addRepresentation("cartoon", { color: "lightgrey" });
          component.autoView();
          changeColorBindSites(
            component,
            tabIndex === 1 ? props.aiPredictionData : filteredData,
            "ball+stick"
          );
        });
    }

    stage.setParameters({ backgroundColor: "white" });
    props.setStage(props.stage); // Remove previous components
  }

  function resetNGLViewerPredictors(stage) {
    resetParameters();
    stage.removeAllComponents();
    stage
      .loadFile(
        pdbFilesPath + "/pdbs/" + props.pdbFolder + "/AF-" + props.pdb + "-F1-model_v4.pdb"
      )
      .then((component) => {
        component.addRepresentation("cartoon", { color: "lightgrey" });
        component.autoView();
        changeColorBindSites(component, props.bindSites);
      });
    stage.setParameters({ backgroundColor: "white" });
    props.setStage(stage); // Remove previous components
  }

  useEffect(() => {
    if (props.puresnetButton && props.graspButton && props.pointsiteButton && props.p2rankButton && props.deeppocketButton && props.stage) {
      resetNGLViewer(props.stage);
    }
  }, [[props.puresnetButton, props.graspButton, props.pointsiteButton]]);

  function resetNGLViewerPopup(stage) {
    resetParameters();
    stage.removeAllComponents();
    stage
      .loadFile(
        pdbFilesPath + "/pdbs/" + props.pdbFolder + "/AF-" + props.pdb + "-F1-model_v4.pdb"
      )
      .then((component) => {
        component.addRepresentation("cartoon", { color: "lightgrey" });
        component.autoView();
        colorAllSitesPopup(component);
        changeColorBindSitesPopup(
          component,
          props.upsetClickResidues,
          bSiteColors[5]
        );
      });
    stage.setParameters({ backgroundColor: "white" });
    props.setStage(stage); // Remove previous components
  }

  function ViewerTabsSummary() {
    return (
      <Box
        sx={{
          borderBottom: 1,
          borderColor: "divider",
          display: "flex",
          justifyContent: "center",
        }}
      >
        <Tabs
          value={props.tabIndex}
          onChange={handleTabChange}
          aria-label="basic tabs example"
          variant="scrollable"
          allowScrollButtonsMobile
        >
          <NoMaxWidthTooltip title="Predictors consensus">
            <Tab
              key={"consensus"}
              sx={{
                "&:hover": {
                  color: "#1976d2",
                  borderBottom: 2,
                  borderColor: "#1976d2",
                },
              }}
              label={"Consensus"}
              {...a11yProps(0)}
            />
          </NoMaxWidthTooltip>
          {props.aiPredictionData.length > 0 ? (
            <NoMaxWidthTooltip title="Meta-predictor based on a machine learning strategy">
              <Tab
                key={"bender-ai"}
                sx={{
                  "&:hover": {
                    color: "#1976d2",
                    borderBottom: 2,
                    borderColor: "#1976d2",
                  },
                }}
                label={"BENDER AI"}
                {...a11yProps(2)}
              />
            </NoMaxWidthTooltip>
          ) : (
            <NoMaxWidthTooltip title="BENDER AI did not predict any binding site residue for this protein">
              <Box>
                <Tab
                  key={"bender-ai"}
                  sx={{
                    "&:hover": {
                      color: "#1976d2",
                      borderBottom: 2,
                      borderColor: "#1976d2",
                    },
                  }}
                  label="BENDER AI"
                  {...a11yProps(2)}
                  disabled
                />
              </Box>
            </NoMaxWidthTooltip>
          )}

          {[...Array(props.numPreds * props.maxConsensusPercent)].map(
            (_, i) => (
              <NoMaxWidthTooltip title={`Predictors convergence in at least ${Math.floor(
                ((props.numPreds * props.maxConsensusPercent - i) /
                  props.numPreds) *
                100
              )}% of results`}>
              <Tab
                key={`${Math.floor(
                  ((props.numPreds * props.maxConsensusPercent - i) /
                    props.numPreds) *
                  100
                )}%`}
                sx={{
                  "&:hover": {
                    color: "#1976d2",
                    borderBottom: 2,
                    borderColor: "#1976d2",
                  },
                }}
                label={`${Math.floor(
                  ((props.numPreds * props.maxConsensusPercent - i) /
                    props.numPreds) *
                  100
                )}%`}
                {...a11yProps(i + 2)}
              />
              </NoMaxWidthTooltip>
            )
          )}
        </Tabs>
      </Box>
    );
  }

  function ViewerTabsPredictors() {
    return (
      <Box
        sx={{
          borderBottom: 1,
          borderColor: "divider",
          display: "flex",
          justifyContent: "center",
        }}
      >
        <Tabs
          value={props.tabIndex}
          onChange={handleTabChange}
          aria-label="basic tabs example"
          variant="scrollable"
          allowScrollButtonsMobile
        >
          {props.bindSites.map((site, i) => (
            <Tab
              label={
                <ColorfulText
                  color={props.bSiteColors[i % props.bSiteColors.length]}
                  hoverColor="grey"
                >
                  Site {i}
                </ColorfulText>
              }
              {...a11yProps(i)}
            />
          ))}
        </Tabs>
      </Box>
    );
  }

  function ViewerTabsPopup() {
    return (
      <Box
        sx={{
          borderBottom: 1,
          borderColor: "divider",
          display: "flex",
          justifyContent: "center",
        }}
      >
        <Tabs
          aria-label="basic tabs example"
          variant="scrollable"
          allowScrollButtonsMobile
          scrollButtons="auto"
          value={0}
        >
          <Tab label={"Intersection"} {...a11yProps(0)} />
        </Tabs>
      </Box>
    );
  }

  function handleBackgroundColor(stage, color) {
    setBGroundColor(color);
    stage.setParameters({ backgroundColor: color });
  }

  function handleRepresentation(stage, repr, tabIndex) {
    const current_pdb = "AF-" + props.pdb + "-F1-model_v4.pdb";
    setProtReprButton(repr);
    if (repr === "surface") {
      stage.getRepresentationsByName("cartoon").dispose();
      stage.getRepresentationsByName("licorice").dispose();
      if (tabIndex === 0 && props.type === "summary") {
        stage.getComponentsByName(current_pdb).addRepresentation(repr, {
          opacity: 0.3,
          colorScheme: "bfactor",
          colorScale: "RdYlBu", // Defines a color scale from red to blue
          colorReverse: true,
        });
      } else {
        stage
          .getComponentsByName(current_pdb)
          .addRepresentation(repr, { opacity: 0.3, color: "papayawhip" });
      }
    } else if (repr === "cartoon") {
      stage.getRepresentationsByName("surface").dispose();
      stage.getRepresentationsByName("licorice").dispose();
      if (tabIndex === 0 && props.type === "summary") {
        stage.getComponentsByName(current_pdb).addRepresentation(repr, {
          colorScheme: "bfactor",
          colorScale: "RdYlBu", // Defines a color scale from red to blue
          colorReverse: true, // Reverses the color scale to use blue for low bfactor values and red for high bfactor values
        });
      } else {
        stage
          .getComponentsByName(current_pdb)
          .addRepresentation(repr, { color: "lightgrey" });
      }
    } else if (repr === "licorice") {
      stage.getRepresentationsByName("cartoon").dispose();
      stage.getRepresentationsByName("surface").dispose();
      if (tabIndex === 0) {
        stage.getComponentsByName(current_pdb).addRepresentation(repr, {
          colorScheme: "bfactor",
          colorScale: "RdYlBu", // Defines a color scale from red to blue
          colorReverse: true, // Reverses the color scale to use blue for low bfactor values and red for high bfactor values
        });
      } else {
        stage.getComponentsByName(current_pdb).addRepresentation(repr);
      }
    } else if (repr === "surface+cartoon") {
      stage.getRepresentationsByName("surface").dispose();
      stage.getRepresentationsByName("licorice").dispose();
      if (tabIndex === 0 && props.type === "summary") {
        stage.getComponentsByName(current_pdb).addRepresentation("cartoon", {
          colorScheme: "bfactor",
          colorScale: "RdYlBu", // Defines a color scale from red to blue
          colorReverse: true, // Reverses the color scale to use blue for low bfactor values and red for high bfactor values
        });
        stage
          .getComponentsByName(current_pdb)
          .addRepresentation("surface", { opacity: 0.3, color: "papayawhip" });
      } else {
        stage
          .getComponentsByName(current_pdb)
          .addRepresentation("cartoon", { color: "lightgrey" });
        stage
          .getComponentsByName(current_pdb)
          .addRepresentation("surface", { opacity: 0.3, color: "papayawhip" });
      }
    }
  }

  function handleBSiteRepresentation(stage, repr, tabIndex) {
    if (tabIndex === 0) return;
    setSiteProtReprButton(repr);
    const current_pdb = "AF-" + props.pdb + "-F1-model_v4.pdb";
    const filteredData = props.consensusData.filter(
      (p) => p[3] >= (props.numPreds - tabIndex + 1) / props.numPreds
    );
    const bSiteString = generateBindSiteStringSummary(filteredData);
    stage.getRepresentationsByName("licorice").dispose();
    stage.getRepresentationsByName("ball+stick").dispose();
    stage.getComponentsByName(current_pdb).addRepresentation("licorice", {
      color: "blue",
      sele: bSiteString,
    });
  }

  function a11yProps(index) {
    return {
      id: `simple-tab-${index}`,
      "aria-controls": `simple-tabpanel-${index}`,
    };
  }

  function resetParameters() {
    setProtReprButton("cartoon");
    setSiteProtReprButton("licorice");
    setBGroundColor("white");
  }

  function handleTabChange(event, newValue) {
    props.setTabIndex(newValue);
    if (props.type === "summary") {
      resetNGLViewer(props.stage, newValue);
    }
    setProtReprButton("cartoon"); // Reset protReprButton to 'cartoon' or your default value
    setSiteProtReprButton("licorice");
    setBGroundColor("white");
  }

  CustomTabPanel.propTypes = {
    children: PropTypes.node,
    index: PropTypes.number.isRequired,
    tabIndex: PropTypes.number.isRequired,
  };

  function CustomTabPanel(props) {
    const { children, value, index, ...other } = props;

    return (
      <div
        role="tabpanel"
        hidden={value !== index}
        id={`simple-tabpanel-${index}`}
        aria-labelledby={`simple-tab-${index}`}
        {...other}
      >
        {value === index && (
          <Box sx={{ p: 0 }}>
            <Typography>{children}</Typography>
          </Box>
        )}
      </div>
    );
  }

  function handleClickOpen() {
    setOpen(true);
  }

  function handleClose(event, reason) {
    if (reason !== "backdropClick") {
      setOpen(false);
    }
  }

  return (
    <div className="col-md-8">
      <Card variant="outlined">
        <Box sx={{ p: 2 }}>
          <Stack
            direction="row"
            justifyContent="space-between"
            alignItems="center"
          >
            <Typography gutterBottom variant="h5" component="div">
              Molecular visualization
            </Typography>
            <NoMaxWidthTooltip title="Download PyMol session">
              <Button
                size="small"
                aria-label="download"
                onClick={() => handleDownloadPymol(props.pdb)}
                variant="outlined"
                startIcon={<DownloadingIcon />}
              >
                PyMol
              </Button>
            </NoMaxWidthTooltip>

          </Stack>
          <Typography color="text.secondary" variant="body2">
            {props.pdb} protein structure along with highlighted binding site
            residues
          </Typography>
        </Box>

        {setViewerTabs()}

        <div className="row">
          <div className="col-md-12">
            <div
              style={{
                position: "relative",
                width: "100%",
                height: "673px",
              }}
            >
              <div
                id={props.type === "popup" ? "viewport-pop" : "viewport"}
                style={{ width: "100%", height: "100%" }}
              ></div>
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  left: "50%",
                  transform: "translateX(-50%)",
                  zIndex: 1,
                }}
              >
                <Box
                  sx={{
                    display: "flex",
                    justifyContent: "center",
                  }}
                >
                  <Stack direction="row" spacing={2}>
                    <div>
                      <NoMaxWidthTooltip title="Visualization settings">
                        <IconButton
                          onClick={handleClickOpen}
                        >
                          <SettingsIcon
                            htmlColor={bgroundColor === "white" ? "" : "white"}
                          />
                        </IconButton>
                      </NoMaxWidthTooltip>
                      <Dialog
                        disableEscapeKeyDown
                        open={open}
                        onClose={handleClose}
                      >
                        <DialogTitle>Visualization settings</DialogTitle>
                        <DialogContent>
                          <Typography color="text.secondary" variant="body2">
                            Use select buttons bellow to change background color
                            and protein representation
                          </Typography>
                          <IconButton
                            aria-label="close"
                            onClick={handleClose}
                            sx={{
                              position: "absolute",
                              right: 8,
                              top: 8,
                              color: (theme) => theme.palette.grey[500],
                            }}
                          >
                            <CloseIcon />
                          </IconButton>
                        </DialogContent>
                        <Divider />
                        <DialogContent>
                          <Box
                            component="form"
                            sx={{ display: "flex", flexWrap: "wrap" }}
                          >
                            <FormControl
                              sx={{ m: 1, maxWidth: 181 }}
                              size="small"
                            >
                              <FormHelperText sx={{ marginLeft: 0 }}>
                                Background color
                              </FormHelperText>
                              <Select
                                labelId="bground-select-small-label"
                                id="bground-select-small"
                                value={bgroundColor}
                                onChange={(e) =>
                                  handleBackgroundColor(
                                    props.stage,
                                    e.target.value
                                  )
                                }
                              >
                                <MenuItem value="black">Black</MenuItem>
                                <MenuItem value="white">White</MenuItem>
                              </Select>
                            </FormControl>
                            <FormControl
                              sx={{ m: 1, maxWidth: 181 }}
                              size="small"
                            >
                              <FormHelperText sx={{ marginLeft: 0 }}>
                                Protein representation
                              </FormHelperText>
                              <Select
                                labelId="prot-select-small-label"
                                id="prot-select-small"
                                value={protReprButton}
                                onChange={(e) =>
                                  handleRepresentation(
                                    props.stage,
                                    e.target.value,
                                    props.tabIndex
                                  )
                                }
                              >
                                <MenuItem value="cartoon">Cartoon</MenuItem>
                                <MenuItem value="licorice">Licorice</MenuItem>
                                <MenuItem value="surface">Surface 1</MenuItem>
                                <MenuItem value="surface+cartoon">
                                  Surface 2
                                </MenuItem>
                              </Select>
                            </FormControl>
                            {/*}
                            {props.tabIndex !== 0 ? (
                              <FormControl
                                sx={{ m: 1, maxWidth: 181 }}
                                size="small"
                              >
                                <FormHelperText sx={{ marginLeft: 0 }}>
                                  Binding site representation
                                </FormHelperText>
                                <Select
                                  labelId="site-select-small-label"
                                  id="site-select-small"
                                  value={siteReprButton}
                                  onChange={(e) =>
                                    handleBSiteRepresentation(
                                      props.stage,
                                      e.target.value,
                                      props.tabIndex
                                    )
                                  }
                                >
                                  <MenuItem value="cartoon">Cartoon</MenuItem>
                                  <MenuItem value="licorice">Licorice</MenuItem>
                                  <MenuItem value="surface">Surface</MenuItem>
                                </Select>
                              </FormControl>
                            ) : null}
                              {*/}
                          </Box>
                        </DialogContent>
                        <DialogActions>
                          <Button onClick={handleClose}>Close</Button>
                        </DialogActions>
                      </Dialog>
                    </div>
                    <MouseHelpPopup bgroundColor={bgroundColor} />
                    <NoMaxWidthTooltip title="Reset visualization">
                      <IconButton
                        aria-label="restart"
                        onClick={() =>
                          resetNGLViewer(props.stage, props.tabIndex)
                        }
                      >
                        <RestartAltIcon
                          htmlColor={bgroundColor === "white" ? "" : "white"}
                        />
                      </IconButton>
                    </NoMaxWidthTooltip>

                  </Stack>
                </Box>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}
