import React, { useState } from "react";
import {
  Button,
  ButtonGroup,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  Box,
  Stack,
  useMediaQuery,
} from "@mui/material";
import Divider from "@mui/material/Divider";

import MolViewerPopup from "../utils/MolViewerPopup";
import CloseIcon from "@mui/icons-material/Close";
import IconButton from "@mui/material/Button";
import { useTheme } from "@mui/material/styles";
import Tooltip, { tooltipClasses } from '@mui/material/Tooltip';
import { styled } from '@mui/system';

const NoMaxWidthTooltip = styled(({ className, ...props }) => (
  <Tooltip {...props} classes={{ popper: className }} arrow />
))({
  [`& .${tooltipClasses.tooltip}`]: {
    maxWidth: 'none',
  },
});

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

const ResponsiveButtonGroup = (props) => {
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down("sm"));
  return (
    <div style={{ overflowX: "auto", display: "flex", flexWrap: "nowrap" }}>
      <ButtonGroup
        sx={{
          display: "flex",
          flexDirection: "row",
          flexWrap: "nowrap",
        }}
      >
        <Button
          variant="contained"
          sx={{
            "&.Mui-disabled": {
              backgroundColor: bSiteColors[5],
              color: "white",
            },
            minWidth: isSmallScreen ? "80px" : "auto",
            flexShrink: 0,
          }}
          disabled
        >
          Intersection
        </Button>
        <NoMaxWidthTooltip title="Click here to show/hide other residues (besides the intersection) of this predictor's results in the molecular viewer">
        <Button
          variant="contained"
          sx={{
            backgroundColor: (theme) => theme.palette.action.disabledBackground,
            "&:not(:disabled)": {
              backgroundColor: bSiteColors[0],
              color: "white",
            },
            minWidth: isSmallScreen ? "80px" : "auto",
            flexShrink: 0,
          }}
          disabled={
            Object.values(props.predsToShow).includes("GRaSP") ? false : true
          }
          onClick={
            props.graspButton === "selected"
              ? () => props.setGraspButton("not-selected")
              : () => props.setGraspButton("selected")
          }
        >
          GRaSP
        </Button>   
        </NoMaxWidthTooltip>
        <NoMaxWidthTooltip title="Click here to show/hide other residues (besides the intersection) of this predictor's results in the molecular viewer">
        <Button
          variant="contained"
          sx={{
            backgroundColor: (theme) => theme.palette.action.disabledBackground,
            "&:not(:disabled)": {
              backgroundColor: bSiteColors[1],
              color: "white",
            },
            minWidth: isSmallScreen ? "80px" : "auto",
            flexShrink: 0,
          }}
          disabled={
            Object.values(props.predsToShow).includes("PUResNet") ? false : true
          }
          onClick={
            props.puresnetButton === "selected"
              ? () => props.setPuresnetButton("not-selected")
              : () => props.setPuresnetButton("selected")
          }
        >
          PUResNet
        </Button>
        </NoMaxWidthTooltip>
        <NoMaxWidthTooltip title="Click here to show/hide other residues (besides the intersection) of this predictor's results in the molecular viewer">
        <Button
          variant="contained"
          sx={{
            backgroundColor: (theme) => theme.palette.action.disabledBackground,
            "&:not(:disabled)": {
              backgroundColor: bSiteColors[3],
              color: "white",
            },
            minWidth: isSmallScreen ? "80px" : "auto",
            flexShrink: 0,
          }}
          disabled={
            Object.values(props.predsToShow).includes("DeepPocket")
              ? false
              : true
          }
          onClick={
            props.deeppocketButton === "selected"
              ? () => props.setDeeppocketButton("not-selected")
              : () => props.setDeeppocketButton("selected")
          }
        >
          DeepPocket
        </Button>
        </NoMaxWidthTooltip>
        <NoMaxWidthTooltip title="Click here to show/hide other residues (besides the intersection) of this predictor's results in the molecular viewer">
        <Button
          variant="contained"
          sx={{
            backgroundColor: (theme) => theme.palette.action.disabledBackground,
            "&:not(:disabled)": {
              backgroundColor: bSiteColors[4],
              color: "white",
            },
            minWidth: isSmallScreen ? "80px" : "auto",
            flexShrink: 0,
          }}
          disabled={
            Object.values(props.predsToShow).includes("PointSite")
              ? false
              : true
          }
          onClick={
            props.pointsiteButton === "selected"
              ? () => props.setPointsiteButton("not-selected")
              : () => props.setPointsiteButton("selected")
          }
        >
          PointSite
        </Button>
        </NoMaxWidthTooltip>
        <NoMaxWidthTooltip title="Click here to show/hide other residues (besides the intersection) of this predictor's results in the molecular viewer">
          <Button
            variant="contained"
            sx={{
              backgroundColor: (theme) => theme.palette.action.disabledBackground,
              "&:not(:disabled)": {
                backgroundColor: "pink",
                color: "white",
              },
              minWidth: isSmallScreen ? "80px" : "auto",
              flexShrink: 0,
            }}
            disabled={
              Object.values(props.predsToShow).includes("P2Rank") ? false : true
            }
            onClick={
              props.p2rankButton === "selected"
                ? () => props.setp2rankButton("not-selected")
                : () => props.setp2rankButton("selected")
            }
          >
            P2Rank
          </Button>
        </NoMaxWidthTooltip>
      </ButtonGroup>
    </div>
  );
};

export default function SummaryPopup(props) {
  const [openInters, setOpenInters] = useState(false);
  const [puresnetButton, setPuresnetButton] = useState("selected");
  const [pointsiteButton, setPointsiteButton] = useState("selected");
  const [graspButton, setGraspButton] = useState("selected");
  const [p2rankButton, setp2rankButton] = useState("selected");
  const [deeppocketButton, setDeeppocketButton] = useState("selected");

  function handleClickOpenInters() {
    setOpenInters(true);
  }

  function handleCloseInters(event, reason) {
    if (reason !== "backdropClick") {
      setOpenInters(false);
    }
  }

  return (
    <>
      <div>
        <Button
          onClick={handleClickOpenInters}
          variant="contained"
          color="success"
        >
          VIEW ON PROTEIN
        </Button>
        <Dialog
          disableEscapeKeyDown
          open={openInters}
          onClose={handleCloseInters}
          maxWidth="lg"
          fullWidth
        >
          <DialogTitle>
            <Box
              sx={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                textAlign: "center",
                width: "100%", // Ensure the box takes full width
              }}
            >
              <Typography gutterBottom variant="h5" component="div">
                Binding site intersection between{" "}
                {props.predsToShow.length === 1
                  ? props.predsToShow[0]
                  : props.predsToShow.slice(0, -1).join(", ") +
                  " and " +
                  props.predsToShow[props.predsToShow.length - 1]}
              </Typography>
              <Typography color="text.secondary" variant="body2">
                Residue colors are displayed according to each predictor.
                Intersection residues are shown in light blue.
              </Typography>
              <Box sx={{ width: "100%", overflowX: "auto", marginTop: 2 }}>
                <Stack
                  direction="row"
                  justifyContent="center" // Align buttons to the center
                  spacing={2} // Add space between the buttons
                  alignItems="center"
                >
                  <ResponsiveButtonGroup
                    predsToShow={props.predsToShow}
                    puresnetButton={puresnetButton}
                    setPuresnetButton={setPuresnetButton}
                    graspButton={graspButton}
                    setGraspButton={setGraspButton}
                    pointsiteButton={pointsiteButton}
                    setPointsiteButton={setPointsiteButton}
                    p2rankButton={p2rankButton}
                    setp2rankButton={setp2rankButton}
                    deeppocketButton={deeppocketButton}
                    setDeeppocketButton={setDeeppocketButton}
                  />
                </Stack>
              </Box>
            </Box>
          </DialogTitle>
          <IconButton
            aria-label="close"
            onClick={handleCloseInters}
            sx={{
              position: "absolute",
              right: 8,
              top: 8,
              color: (theme) => theme.palette.grey[500],
            }}
          >
            <CloseIcon />
          </IconButton>
          <Divider />
          <DialogContent>
            <MolViewerPopup
              pdb={props.pdb}
              pdbFolder={props.pdbFolder}
              bindingResidues={props.bindingResidues}
              numPreds={props.numPreds}
              consensusData={props.consensusData}
              bindSites={props.bindSites}
              graspSites={props.graspSites}
              puresnetSites={props.puresnetSites}
              g
              deeppocketSites={props.deeppocketSites}
              pointsiteSites={props.pointsiteSites}
              p2rankSites={props.p2rankSites}
              predsToShow={props.predsToShow}
              upsetClickResidues={props.upsetClickResidues}
              puresnetButton={puresnetButton}
              graspButton={graspButton}
              pointsiteButton={pointsiteButton}
              p2rankButton={p2rankButton}
              setp2rankButton={setp2rankButton}
              deeppocketButton={deeppocketButton}
              setDeeppocketButton={setDeeppocketButton}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseInters}>Close</Button>
          </DialogActions>
        </Dialog>
      </div>
    </>
  );
}
