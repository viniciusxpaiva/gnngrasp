import React, { useState } from "react";
import Stack from "@mui/material/Stack";
import IconButton from "@mui/material/IconButton";
import RemoveRedEyeOutlinedIcon from "@mui/icons-material/RemoveRedEyeOutlined";
import PropTypes from "prop-types";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import Card from "@mui/material/Card";
import Divider from "@mui/material/Divider";
import { styled } from "@mui/material/styles";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell, { tableCellClasses } from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Paper from "@mui/material/Paper";
import Button from "@mui/material/Button";
import DownloadingIcon from "@mui/icons-material/Downloading";
import Tooltip, { tooltipClasses } from '@mui/material/Tooltip';


//const csvDLUrl = "https://benderdb.ufv.br/benderdb-data/results/"
const csvDLUrl = process.env.PUBLIC_URL + "/results/"

const NoMaxWidthTooltip = styled(({ className, ...props }) => (
  <Tooltip {...props} classes={{ popper: className }} arrow />
))({
  [`& .${tooltipClasses.tooltip}`]: {
    maxWidth: 'none',
  },
});

const StyledTableCell = styled(TableCell)(({ theme }) => ({
  [`&.${tableCellClasses.head}`]: {
    backgroundColor: "grey",
    color: theme.palette.common.white,
    height: 50,
  },
  [`&.${tableCellClasses.body}`]: {
    fontSize: 15,
  },
}));

const StyledTableRow = styled(TableRow)(({ theme }) => ({
  "&:nth-of-type(odd)": {
    backgroundColor: theme.palette.action.hover,
  },
  // hide last border
  "&:last-child td, &:last-child th": {
    border: 0,
  },
}));

export default function ResiduesTabs(props) {

  const [previousFocusRes, setPreviousFocusRes] = useState("");

  function ContentTabs() {
    if (props.type === "summary") {
      return <ContentTabsSummary />;
    } else if (props.type === "predictors") {
      return <ContentTabsPredictors />;
    } else if (props.type === "popup") {
      return <ContentTabsPopup />; // Or handle other cases
    } else {
      return null;
    }
  }

  function handleDownloadResults(predictor, protName) {
    const fileUrl = csvDLUrl + protName + "_" + predictor + "_results.csv";
    const link = document.createElement("a");
    link.href = fileUrl;
    link.download = protName + "_" + predictor + "_results.csv";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function ContentTabsSummary() {
    return (
      <Card variant="outlined" sx={{ marginTop: { xs: 2, md: 0 } }}>
        <Box sx={{ p: 2, height: 137 }}>
          <Stack
            direction="row"
            justifyContent="space-between"
            alignItems="center"
          >
            <Typography gutterBottom variant="h5" component="div">
              Binding site residues
            </Typography>
          </Stack>
          <Typography color="text.secondary" variant="body2">
            {props.tabIndex === 0
              ? "The colder the tones, the fewer residues are predicted as binding sites in that region. The warmer the tones, the more residues are predicted in that region."
              : props.tabIndex === 1
                ? "Residues displayed below are predicted by BENDER AI, a meta-predictor based on a machine learning strategy."
                : `Residues displayed below are presented in at least ${Math.floor(((props.numPreds * props.maxConsensusPercent - props.tabIndex + 2) / props.numPreds) * 100)}% of predictors results.`}

          </Typography>
        </Box>
        <Divider />
        <Box sx={{ p: 0 }}>
          <Box sx={{ width: "100%" }}>
            <CustomTabPanel value={props.tabIndex} index={0}>
              <TableContainer component={Paper} sx={{ height: 676 }}>
                <Table stickyHeader aria-label="customized table" size="small">
                  <TableHead>
                    <TableRow>
                      <StyledTableCell align="center">Residue</StyledTableCell>
                      <StyledTableCell align="center">Number</StyledTableCell>
                      <StyledTableCell align="center">Chain</StyledTableCell>
                      <StyledTableCell align="center">Look at</StyledTableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {props.consensusData.map((p, i) => (
                      <StyledTableRow key={i}>
                        <StyledTableCell align="center">
                          {p[1]}
                        </StyledTableCell>
                        <StyledTableCell align="center">
                          {p[2]}
                        </StyledTableCell>
                        <StyledTableCell align="center">
                          {p[0]}
                        </StyledTableCell>
                        <StyledTableCell align="center">
                          <NoMaxWidthTooltip title="Focus on this residue">
                            <IconButton
                              className="p-1"
                              aria-label="focus-res"
                              onClick={() =>
                                focusResidue(props.stage, p[2], p[0])
                              }
                            >
                              <RemoveRedEyeOutlinedIcon />
                            </IconButton>
                          </NoMaxWidthTooltip>
                        </StyledTableCell>
                      </StyledTableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CustomTabPanel>
            <CustomTabPanel value={props.tabIndex} index={1}>
              <TableContainer component={Paper} sx={{ height: 676 }}>
                <Table stickyHeader aria-label="customized table" size="small">
                  <TableHead>
                    <TableRow>
                      <StyledTableCell align="center">Residue</StyledTableCell>
                      <StyledTableCell align="center">Number</StyledTableCell>
                      <StyledTableCell align="center">Chain</StyledTableCell>
                      <StyledTableCell align="center">Look at</StyledTableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {props.aiPredictionData.map((p, i) => (
                      <StyledTableRow key={i}>
                        <StyledTableCell align="center">
                          {p[1]}
                        </StyledTableCell>
                        <StyledTableCell align="center">
                          {p[2]}
                        </StyledTableCell>
                        <StyledTableCell align="center">
                          {p[0]}
                        </StyledTableCell>
                        <StyledTableCell align="center">
                          <NoMaxWidthTooltip title="Focus on this residue">
                            <IconButton
                              className="p-1"
                              aria-label="focus-res"
                              onClick={() =>
                                focusResidue(props.stage, p[2], p[0])
                              }
                            >
                              <RemoveRedEyeOutlinedIcon />
                            </IconButton>
                          </NoMaxWidthTooltip>
                        </StyledTableCell>
                      </StyledTableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CustomTabPanel>
            {[...Array(props.numPreds)].map((_, i) => (
              <CustomTabPanel value={props.tabIndex} index={i + 2}>
                <TableContainer component={Paper} sx={{ height: 676 }}>
                  <Table
                    stickyHeader
                    aria-label="customized table"
                    size="small"
                  >
                    <TableHead>
                      <TableRow>
                        <StyledTableCell align="center">
                          Residue
                        </StyledTableCell>
                        <StyledTableCell align="center">Number</StyledTableCell>
                        <StyledTableCell align="center">Chain</StyledTableCell>
                        <StyledTableCell align="center">
                          Look at
                        </StyledTableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {props.consensusData.map((p, j) => {
                        if (p[3] >= (props.numPreds * props.maxConsensusPercent - i) / props.numPreds) {
                          return (
                            <StyledTableRow key={i}>
                              <StyledTableCell align="center">
                                {p[1]}
                              </StyledTableCell>
                              <StyledTableCell align="center">
                                {p[2]}
                              </StyledTableCell>
                              <StyledTableCell align="center">
                                {p[0]}
                              </StyledTableCell>
                              <StyledTableCell align="center">
                                <NoMaxWidthTooltip title="Focus on this residue">
                                  <IconButton
                                    className="p-1"
                                    aria-label="focus-res"
                                    onClick={() =>
                                      focusResidue(props.stage, p[2], p[0])
                                    }
                                  >
                                    <RemoveRedEyeOutlinedIcon />
                                  </IconButton>
                                </NoMaxWidthTooltip>
                              </StyledTableCell>
                            </StyledTableRow>
                          );
                        }
                        return null;
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CustomTabPanel>
            ))}
          </Box>
        </Box>
      </Card>
    );
  }

  function ContentTabsPredictors() {
    return (
      <Card variant="outlined" sx={{ marginTop: { xs: 2, md: 0 } }}>
        <Box sx={{ p: 2, height: 137 }}>
          <Stack
            direction="row"
            justifyContent="space-between"
            alignItems="center"
          >
            <Typography gutterBottom variant="h5" component="div">
              <span className="align-middle">{props.pred + " sites"}</span>
            </Typography>
            <NoMaxWidthTooltip title="Download results">
              <Button
                size="small"
                aria-label="download"
                //title="Download results"
                onClick={() => handleDownloadResults(props.pred, props.pdb)}
                variant="outlined"
                startIcon={<DownloadingIcon />}
                sx={{
                  height: "40px", // Set the height to match the IconButton's height
                }}
              >
                Results
              </Button>
            </NoMaxWidthTooltip>

          </Stack>
          <Typography color="text.secondary" variant="body2">
            List of binding site residues.
          </Typography>
        </Box>
        <Divider />
        <Box sx={{ p: 0 }}>
          <Box sx={{ width: "100%" }}>
            {props.bindSites.sort(
              (a, b) => parseInt(a.number) - parseInt(b.number)
            ).map((p, i) => (
              <CustomTabPanel value={props.tabIndex} index={i}>
                <TableContainer component={Paper} sx={{ height: 676 }}>
                  <Table
                    stickyHeader
                    aria-label="customized table"
                    size="small"
                  >
                    <TableHead>
                      <TableRow>
                        <StyledTableCell align="center">
                          Residue
                        </StyledTableCell>
                        <StyledTableCell align="center">Number</StyledTableCell>
                        <StyledTableCell align="center">Chain</StyledTableCell>
                        <StyledTableCell align="center">
                          Look at
                        </StyledTableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {p.map((res, j) => (
                        <StyledTableRow key={i}>
                          <StyledTableCell align="center">
                            {res[1]}
                          </StyledTableCell>
                          <StyledTableCell align="center">
                            {res[2]}
                          </StyledTableCell>
                          <StyledTableCell align="center">
                            {res[0]}
                          </StyledTableCell>
                          <StyledTableCell align="center">
                            <NoMaxWidthTooltip title="Focus on this residue">
                              <IconButton
                                className="p-1"
                                aria-label="focus-res"
                                onClick={() =>
                                  focusResidue(props.stage, res[2], res[0])
                                }
                              >
                                <RemoveRedEyeOutlinedIcon />
                              </IconButton>
                            </NoMaxWidthTooltip>
                          </StyledTableCell>
                        </StyledTableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CustomTabPanel>
            ))}
          </Box>
        </Box>
      </Card>
    );
  }

  function ContentTabsPopup() {
    return (
      <Card variant="outlined" sx={{ marginTop: { xs: 2, md: 0 } }}>
        <Box sx={{ p: 2, height: 137 }}>
          <Stack
            direction="row"
            justifyContent="space-between"
            alignItems="center"
          >
            <Typography gutterBottom variant="h5" component="div">
              Binding site residues
            </Typography>
          </Stack>
          <Typography color="text.secondary" variant="body2">
            Residues in the selected intersection are listed below.
          </Typography>
        </Box>
        <Divider />
        <Box sx={{ p: 0 }}>
          <Box sx={{ width: "100%" }}>
            <CustomTabPanel value={props.tabIndex} index={0}>
              <TableContainer component={Paper} sx={{ height: 676 }}>
                <Table stickyHeader aria-label="customized table" size="small">
                  <TableHead>
                    <TableRow>
                      <StyledTableCell align="center">Residue</StyledTableCell>
                      <StyledTableCell align="center">Number</StyledTableCell>
                      <StyledTableCell align="center">Chain</StyledTableCell>
                      <StyledTableCell align="center">Look at</StyledTableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {props.upsetClickResidues.map((res, i) => (
                      <StyledTableRow key={i}>
                        <StyledTableCell align="center">
                          {res.split("-")[0]}
                        </StyledTableCell>
                        <StyledTableCell align="center">
                          {res.split("-")[1]}
                        </StyledTableCell>
                        <StyledTableCell align="center">
                          {res.split("-")[2]}
                        </StyledTableCell>
                        <StyledTableCell align="center">
                          <NoMaxWidthTooltip title="Focus on this residue">
                            <IconButton
                              className="p-1"
                              aria-label="focus-res"
                              onClick={() =>
                                focusResidue(
                                  props.stage,
                                  res.split("-")[1],
                                  res.split("-")[2]
                                )
                              }
                            >
                              <RemoveRedEyeOutlinedIcon />
                            </IconButton>
                          </NoMaxWidthTooltip>

                        </StyledTableCell>
                      </StyledTableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CustomTabPanel>
          </Box>
        </Box>
      </Card>
    );
  }

  function focusResidue(stage, resNum, chain) {
    const sele = resNum + ":" + chain;
    if (previousFocusRes === sele) {
      stage.getRepresentationsByName("surface").dispose();
      setPreviousFocusRes("");
      return;
    }
    const pdb_id = "input.pdb";
    //stage.getRepresentationsByName("surface").dispose();
    stage.getComponentsByName(pdb_id).addRepresentation("surface", {
      sele: sele,
      opacity: 0.5,
      side: "front",
    });
    stage.getComponentsByName(pdb_id).autoView(sele);
    setPreviousFocusRes(sele);
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
  return <div className="col-md-4">{ContentTabs()}</div>;
}
