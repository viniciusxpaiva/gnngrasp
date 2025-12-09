import React, { useState } from "react";
import { Link } from "react-router-dom";
import PropTypes from "prop-types";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import MolViewerSummary from "../utils/MolViewerSummary";
import MolViewerPredictors from "../utils/MolViewerPredictors";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import Stack from "@mui/material/Stack";
import { MDBDataTable } from "mdbreact";
import UpsetPlot from "../visualization/UpsetPlot";
import "reactjs-popup/dist/index.css";
import Card from "@mui/material/Card";
import Divider from "@mui/material/Divider";
import SummaryPopup from "./SummaryPopup";
import Tooltip, { tooltipClasses } from '@mui/material/Tooltip';
import { styled } from '@mui/system';

const NoMaxWidthTooltip = styled(({ className, ...props }) => (
  <Tooltip {...props} classes={{ popper: className }} arrow/>
))({
  [`& .${tooltipClasses.tooltip}`]: {
    maxWidth: 'none',
  },
});

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

function a11yProps(index) {
  return {
    id: `simple-tab-${index}`,
    "aria-controls": `simple-tabpanel-${index}`,
  };
}

export default function ResultsPageTabs(props) {
  const [value, setValue] = useState(0);
  const handleChange = (event, newValue) => {
    setValue(newValue);
  };

  return (
    <>
      <Box sx={{ width: "100%" }}>
        <CustomTabPanel value={value} index={0}>
          <MolViewerPredictors
            pred={props.predictors[0]}
            predictors={props.predictors}
            activeTab={props.predictorTab}
            pdb={props.pdb}
            bindSites={props.graspSites}
            pdbFolder={props.pdbFolder}
            proteinFullName={props.proteinFullName}
          />
        </CustomTabPanel>
      </Box>
    </>
  );
}
