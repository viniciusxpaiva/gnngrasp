import React, { useState } from "react";
import PropTypes from "prop-types";
import BaseLayout from "../components/layout/base";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Divider from "@mui/material/Divider";
import Stack from "@mui/material/Stack";
import { Link } from 'react-router-dom';

const Help = () => {
  const [value, setValue] = useState(0);
  function a11yProps(index) {
    return {
      id: `simple-tab-${index}`,
      "aria-controls": `simple-tabpanel-${index}`,
    };
  }

  const handleChange = (event, newValue) => {
    setValue(newValue);
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
          <Box sx={{ p: 3 }}>
            <Typography>{children}</Typography>
          </Box>
        )}
      </div>
    );
  }

  CustomTabPanel.propTypes = {
    children: PropTypes.node,
    index: PropTypes.number.isRequired,
    value: PropTypes.number.isRequired,
  };
  return (
    <BaseLayout>
      <div
        className="container-fluid bg-light-dark text-white mt-0 py-4"
        id="help-submit"
      >
        <div className="row justify-content-center">
          <div class="col-md-12 text-center">
            <h6 className="display-6 text-light">How to use BENDER DB</h6>
          </div>
        </div>
      </div>
      <div className="container">
        <Box sx={{ width: "100%" }}>
          <Box
            sx={{
              borderBottom: 1,
              borderColor: "divider",
              display: "flex",
              justifyContent: "center",
            }}
          >
            <Tabs
              value={value}
              onChange={handleChange}
              aria-label="basic tabs example"
              variant="scrollable"
              allowScrollButtonsMobile
            >
              <Tab
                label="Visualizations"
                sx={{
                  "&:hover": {
                    color: "#1976d2",
                    borderBottom: 2,
                    borderColor: "#1976d2",
                  },
                }}
                {...a11yProps(0)}
              />
              <Tab
                label="Available data"
                sx={{
                  "&:hover": {
                    color: "#1976d2",
                    borderBottom: 2,
                    borderColor: "#1976d2",
                  },
                }}
                {...a11yProps(1)}
              />
              <Tab
                label="Submit a protein"
                sx={{
                  "&:hover": {
                    color: "#1976d2",
                    borderBottom: 2,
                    borderColor: "#1976d2",
                  },
                }}
                {...a11yProps(2)}
              />
              <Tab
                label="Results page"
                sx={{
                  "&:hover": {
                    color: "#1976d2",
                    borderBottom: 2,
                    borderColor: "#1976d2",
                  },
                }}
                {...a11yProps(3)}
              />
              <Tab
                label="Summary content"
                sx={{
                  "&:hover": {
                    color: "#1976d2",
                    borderBottom: 2,
                    borderColor: "#1976d2",
                  },
                }}
                {...a11yProps(4)}
              />
              <Tab
                label="Predictor content"
                sx={{
                  "&:hover": {
                    color: "#1976d2",
                    borderBottom: 2,
                    borderColor: "#1976d2",
                  },
                }}
                {...a11yProps(5)}
              />
            </Tabs>
          </Box>
          <CustomTabPanel value={value} index={0}>
            <Card variant="outlined">
              <Box sx={{ p: 2 }}>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Typography gutterBottom variant="h5" component="div">
                    Molecular viewer
                  </Typography>
                </Stack>
                <Typography color="text.secondary" variant="body2">
                  BENDER DB employs NGL Viewer for molecular visualization
                </Typography>
              </Box>
              <Divider />
              <Box sx={{ p: 2 }}>
                <Stack
                  alignItems="center"
                  sx={{ paddingLeft: 10, paddingRight: 10 }}
                >
                  <Typography
                    color="text.secondary"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>1.</b> The molecule viewer displays binding site residues
                    highlighted in the context of the protein structure
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>2.</b> The protein structure visualization can be
                    customized. The buttons within the viewer allow users to
                    change the representation of the protein structure in
                    different styles (cartoon, licorice, and surface) and to
                    adjust the background color.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3 mb-4"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>3.</b> When visualizing the results summary, tabs in the
                    molecular viewer display different graphical representations
                    of the protein binding sites. The Consensus tab presents a
                    heatmap of all binding site residues. Shades of blue
                    indicate a lower or no occurrence of binding site residues
                    calculated by the predictors, while shades of red indicate a
                    higher presence of binding site residues. The BENDER AI tab
                    shows binding site residues predicted by our Artificial
                    Intelligence model. The tabs with percentages show the
                    occurrence of binding site residues calculated by all the
                    predictors. For example, the 80% tab displays all binding
                    site residues in at least 80% of the results.
                  </Typography>

                  <img
                    src="img/help-E1a.png"
                    className="img-fluid"
                    style={{
                      maxWidth: "55%",
                      maxHeight: "100%",
                      width: "auto",
                      height: "auto",
                    }}
                    alt="help-E1"
                  />
                </Stack>
              </Box>
            </Card>
            <Card className="mt-4" variant="outlined">
              <Box sx={{ p: 2 }}>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Typography gutterBottom variant="h5" component="div">
                    UpSet plot
                  </Typography>
                </Stack>
                <Typography color="text.secondary" variant="body2">
                  This comprehensive plot presents all sets and potential
                  intersections between them for binding sites predicted by all
                  tools, providing a thorough overview.
                </Typography>
              </Box>
              <Divider />
              <Box sx={{ p: 2 }}>
                <Stack
                  alignItems="left"
                  sx={{ paddingLeft: 10, paddingRight: 10 }}
                >
                  <Typography
                    color="text.secondary"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>4.</b> Horizontal bars represent the number of binding
                    site residues predicted by each tool.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>5.</b> The central region of this plot (nodes connected
                    by edges) displays all possible intersections between the
                    predictors. The example below shows the intersection between
                    GRaSP and P2Rank, indicating that there are 20 binding site
                    residues on which they agree.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>6.</b> The vertical bars display the size of each
                    selected set/intersection. When a set/intersection is
                    selected, the bar corresponding to the selection turns
                    yellow. In the other bars, which do not correspond to the
                    selection, a yellow region appears, corresponding to the
                    number of selected sites that are present in the other
                    sets/intersections.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3 mb-4"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>7.</b> Upon selecting an intersection, the "View on
                    protein" button appears. Clicking this button opens a window
                    containing a molecular viewer that displays the binding site
                    residues of the selected intersection. For example, only the
                    residues from the GRaSP and P2Rank predictors will be shown
                    here.
                  </Typography>

                  <img
                    src="img/help-E2a.png"
                    className="img-fluid"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      width: "auto",
                      height: "auto",
                    }}
                    alt="help-E2"
                  />
                </Stack>
              </Box>
            </Card>
          </CustomTabPanel>
          <CustomTabPanel value={value} index={1}>
            <Card variant="outlined">
              <Box sx={{ p: 2 }}>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Typography gutterBottom variant="h5" component="div">
                    Available data
                  </Typography>
                </Stack>
                <Typography color="text.secondary" variant="body2">
                  All available data in BENDER DB can be found on the Available
                  Data page
                </Typography>
              </Box>
              <Divider />
              <Box sx={{ p: 2 }}>
                <Stack
                  alignItems="left"
                  sx={{ paddingLeft: 10, paddingRight: 10 }}
                >
                  <Typography
                    color="text.secondary"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>1.</b> The list below shows the ten proteomes of
                    pathogenic agents of neglected diseases available in BENDER
                    DB.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>2.</b> All 101,813 proteins cataloged by BENDER DB are
                    listed in the table.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>3.</b> The results screen for the selected protein is
                    displayed by clicking the highlighted button.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3 mb-4"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>4.</b> Users can search for a specific protein, proteome,
                    or neglected disease.
                  </Typography>

                  <img
                    src="img/help-F1a.png"
                    className="img-fluid"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      width: "auto",
                      height: "auto",
                    }}
                    alt="help-F1"
                  />
                  <img
                    src="img/help-F2a.png"
                    className="img-fluid"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      width: "auto",
                      height: "auto",
                    }}
                    alt="help-F2"
                  />
                </Stack>
              </Box>
            </Card>
          </CustomTabPanel>
          <CustomTabPanel value={value} index={2}>
            <Card variant="outlined">
              <Box sx={{ p: 2 }}>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Typography gutterBottom variant="h5" component="div">
                    Submit a protein
                  </Typography>
                </Stack>
                <Typography color="text.secondary" variant="body2">
                  Input search bar at homepage
                </Typography>
              </Box>
              <Divider />
              <Box sx={{ p: 2 }}>
                <Stack
                  alignItems="left"
                  sx={{ paddingLeft: 10, paddingRight: 10 }}
                >
                  <Typography
                    color="text.secondary"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>1.</b> The search must be performed by entering a valid
                    UniProt accession from a protein in the proteome of a
                    neglected disease pathogen.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3 mb-4"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>2.</b> A complete list of all proteins and proteomes
                    available in BENDER DB can be found on the{" "}
                    <Typography
                      noWrap
                      component={Link}
                      to="/datatable"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Available Data page
                    </Typography>
                    .
                  </Typography>

                  <img
                    src="img/help-A1a.png"
                    className="img-fluid"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      width: "auto",
                      height: "auto",
                    }}
                    alt="help-A1"
                  />
                </Stack>
              </Box>
            </Card>
          </CustomTabPanel>
          <CustomTabPanel value={value} index={3}>
            <Card variant="outlined">
              <Box sx={{ p: 2 }}>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Typography gutterBottom variant="h5" component="div">
                    Results page
                  </Typography>
                </Stack>
                <Typography color="text.secondary" variant="body2">
                  Protein binding site results
                </Typography>
              </Box>
              <Divider />
              <Box sx={{ p: 2 }}>
                <Stack
                  alignItems="left"
                  sx={{ paddingLeft: 10, paddingRight: 10 }}
                >
                  <Typography
                    color="text.secondary"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>1.</b> When searching for a protein, information about
                    binding sites for this specific entry is displayed on the
                    results screen.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>2.</b> Results are presented in the Summary tab, in which
                    general aspects of the protein and its binding site residues
                    can be viewed, bringing together results from different
                    predictors.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>3.</b> Individual results for each binding site predictor
                    can also be accessed.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3 mb-4"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>4.</b> The protein's name and the organism to which it
                    belongs are displayed at the top of the screen.
                  </Typography>

                  <img
                    src="img/help-B1a.png"
                    className="img-fluid"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      width: "auto",
                      height: "auto",
                    }}
                    alt="help-B1"
                  />
                </Stack>
              </Box>
            </Card>
          </CustomTabPanel>
          <CustomTabPanel value={value} index={4}>
            <Card variant="outlined">
              <Box sx={{ p: 2 }}>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Typography gutterBottom variant="h5" component="div">
                    Summary information
                  </Typography>
                </Stack>
                <Typography color="text.secondary" variant="body2">
                  Molecular visualization and binding site residues
                </Typography>
              </Box>
              <Divider />
              <Box sx={{ p: 2 }}>
                <Stack
                  alignItems="left"
                  sx={{ paddingLeft: 10, paddingRight: 10 }}
                >
                  <Typography
                    color="text.secondary"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>1.</b> Structural representation of the protein
                    considered. Tabs show different thresholds of agreement
                    between predictors. For instance, in the Tab 80% are
                    presented residues detected as part of a binding site by at
                    least 80% of the predictors.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>2.</b> PyMOL session download, containing the consensus
                    visualization of predicted binding site residues.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3 mb-4"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>3.</b> On the right-hand side, a table with predicted
                    binding site residues is provided according to the
                    visualization selected in the molecular viewer tabs. The
                    name (3-letter code), number, and chain of each residue are
                    displayed in the table. By clicking on the 'Look at' icon,
                    the selected residue is highlighted in the molecular viewer.
                  </Typography>

                  <img
                    src="img/help-C1a.png"
                    className="img-fluid"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      width: "auto",
                      height: "auto",
                    }}
                    alt="help-C1"
                  />
                </Stack>
              </Box>
            </Card>
            <Card className="mt-4" variant="outlined">
              <Box sx={{ p: 2 }}>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Typography gutterBottom variant="h5" component="div">
                    Binding sites intersections
                  </Typography>
                </Stack>
                <Typography color="text.secondary" variant="body2">
                  UpSet plot for visualizing intersections between predictors
                </Typography>
              </Box>
              <Divider />
              <Box sx={{ p: 2 }}>
                <Stack
                  alignItems="left"
                  sx={{ paddingLeft: 10, paddingRight: 10 }}
                >
                  <Typography
                    color="text.secondary"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>4.</b> The UpSet plot presents all possible sets and
                    intersections of predictor results. The visualization is
                    interactive, allowing the selection of any subset of data.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3 mb-4"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>5.</b> When selecting an intersection of predictors, the
                    "View on protein" button appears, enabling a more in-depth
                    analysis of the residues in the selected subset in a
                    separate window with a molecular viewer.
                  </Typography>

                  <img
                    src="img/help-C2a.png"
                    className="img-fluid"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      width: "auto",
                      height: "auto",
                    }}
                    alt="help-C2"
                  />
                </Stack>
              </Box>
            </Card>
            <Card className="mt-4" variant="outlined">
              <Box sx={{ p: 2 }}>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Typography gutterBottom variant="h5" component="div">
                    Intersection visualization
                  </Typography>
                </Stack>
                <Typography color="text.secondary" variant="body2">
                  Pop-up window to analyze an intersection selected among
                  predictors results
                </Typography>
              </Box>
              <Divider />
              <Box sx={{ p: 2 }}>
                <Stack
                  alignItems="left"
                  sx={{ paddingLeft: 10, paddingRight: 10 }}
                >
                  <Typography
                    color="text.secondary"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>6.</b> The buttons indicate the predictors of the
                    selected intersection. Button colors correspond to the
                    residues in the molecule. By clicking the buttons, users can
                    show/hide the residues of that specific predictor in the
                    structure visualization.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>7.</b> The molecular viewer displays the residues of the
                    selected intersection and the other residues from the
                    selected predictors that do not belong to the intersection.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3 mb-4"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>8.</b> List of residues from the selected intersection.
                  </Typography>

                  <img
                    src="img/help-C3a.png"
                    className="img-fluid"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      width: "auto",
                      height: "auto",
                    }}
                    alt="help-C3"
                  />
                </Stack>
              </Box>
            </Card>
            <Card className="mt-4" variant="outlined">
              <Box sx={{ p: 2 }}>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Typography gutterBottom variant="h5" component="div">
                    Binding site data table
                  </Typography>
                </Stack>
                <Typography color="text.secondary" variant="body2">
                  Interactive table with predictors and their respective binding
                  site residues
                </Typography>
              </Box>
              <Divider />
              <Box sx={{ p: 2 }}>
                <Stack
                  alignItems="left"
                  sx={{ paddingLeft: 10, paddingRight: 10 }}
                >
                  <Typography
                    color="text.secondary"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>9.</b> Overall results of the identified binding site
                    residues are described, providing the total number of sites
                    and residues calculated by the predictors.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3 mb-4"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>10.</b> Interactive table providing easy access to
                    predicted binding site residues and predictors. The
                    occurrence column lists the number of binding sites in which
                    each residue is present.
                  </Typography>

                  <img
                    src="img/help-C4a.png"
                    className="img-fluid"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      width: "auto",
                      height: "auto",
                    }}
                    alt="help-C3"
                  />
                </Stack>
              </Box>
            </Card>
          </CustomTabPanel>
          <CustomTabPanel value={value} index={5}>
            <Card variant="outlined">
              <Box sx={{ p: 2 }}>
                <Stack
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                >
                  <Typography gutterBottom variant="h5" component="div">
                    Predictor data
                  </Typography>
                </Stack>
                <Typography color="text.secondary" variant="body2">
                  Molecular visualization and binding site residues
                </Typography>
              </Box>
              <Divider />
              <Box sx={{ p: 2 }}>
                <Stack
                  alignItems="left"
                  sx={{ paddingLeft: 10, paddingRight: 10 }}
                >
                  <Typography
                    color="text.secondary"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>1.</b> The molecular viewer displays the protein
                    structure, with the binding site residues highlighted and
                    represented as ball and stick. For those tools that group
                    predicted residues in binding sites, colors were used to
                    differentiate between the sites.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>2.</b> The various binding sites detected in the protein
                    appear as tabs in the molecular viewer. The colors
                    correspond to the residues shown in the protein structure.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>3.</b> Download button for the PyMOL session, containing
                    all the identified binding site residues in the protein.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>4.</b> Download button for the individual predictor
                    results in CSV format.
                  </Typography>
                  <Typography
                    color="text.secondary"
                    className="mt-3 mb-4"
                    variant="body1"
                    align="justify"
                    sx={{ width: "100%" }}
                  >
                    <b>5.</b> Table displaying the residues of the selected
                    binding site.
                  </Typography>

                  <img
                    src="img/help-D1a.png"
                    className="img-fluid"
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      width: "auto",
                      height: "auto",
                    }}
                    alt="help-D1"
                  />
                </Stack>
              </Box>
            </Card>
          </CustomTabPanel>
        </Box>
      </div>
    </BaseLayout>
  );
};
export default Help;
