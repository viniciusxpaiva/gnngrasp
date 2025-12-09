import React from "react";
import { Link } from "react-router-dom";
import BaseLayout from "../components/layout/base";
import Stack from "@mui/material/Stack";
import Alert from "@mui/material/Alert";
import AlertTitle from "@mui/material/AlertTitle";
import Button from "@mui/material/Button";
import Box from "@mui/material/Box";

const NotFound = () => {
  return (
    <BaseLayout>
      <div
        className="container-fluid bg-light-dark text-white mt-0 py-4"
        id="help-submit"
      >
        <div className="row justify-content-center">
          <div className="col-md-12 text-center">
            <h6 className="display-6 text-light">
              Requested PDB structure is not available
            </h6>
          </div>
        </div>
      </div>

      <div className="container mt-4">
        <Stack sx={{ width: "100%" }} spacing={2}>
          <Alert variant="outlined" severity="error">
            <AlertTitle>
              <strong>GNN-GRaSP could not retrieve the PDB structure</strong>
            </AlertTitle>
            We were unable to download or process the PDB file for the
            provided input.
            <br />
            <br />
            Please check that:
            <ul>
              <li>The PDB code is valid (4-character identifier, e.g. “4HHB”).</li>
              <li>
                The structure is publicly available in the Protein Data Bank.
              </li>
              <li>
                There are no typos, extra spaces, or special characters in the
                PDB code.
              </li>
            </ul>
            You can also try uploading a local <code>.pdb</code> file directly
            on the home page.
            <br />
            <br />
            <Box display="flex" justifyContent="left" gap={4}>
              <Link to={"/"}>
                <Button variant="contained" color="primary">
                  Back to home page
                </Button>
              </Link>
            </Box>
          </Alert>
        </Stack>
      </div>
    </BaseLayout>
  );
};

export default NotFound;
