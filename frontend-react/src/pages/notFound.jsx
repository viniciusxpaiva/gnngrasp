import BaseLayout from "../components/layout/base";
import { Link } from "react-router-dom";
import { Box, Alert, AlertTitle, Stack, Button } from "@mui/material";

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
              Searched protein is not available in BENDER DB
            </h6>
          </div>
        </div>
      </div>
      <div className="container mt-4">
        <Stack sx={{ width: "100%" }} spacing={2}>
          <Alert variant="outlined" severity="error">
            <AlertTitle>
              <strong>BENDER DB could not find input protein</strong>
            </AlertTitle>
            Please inform a valid UniProt accession from neglected disease
            proteome.
            <br></br>
            <br></br>
            The following table lists all available proteins in BENDER DB. Use
            the search bar to determine if a protein is available in database.
            <br></br>
            <br></br>
            <Box display="flex" justifyContent="left" gap={4}>
              <Link to={"/"}>
                <Button variant="contained" color="primary">
                  Go to home page
                </Button>
              </Link>
              <Link to={"/datatable"}>
                <Button variant="contained" color="primary">
                  Go to data table
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
