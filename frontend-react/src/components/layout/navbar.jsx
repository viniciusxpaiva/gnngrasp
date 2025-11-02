import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Container from "@mui/material/Container";
import SearchIcon from "@mui/icons-material/Search";
import InputBase from "@mui/material/InputBase";
import { styled, alpha } from "@mui/material/styles";
import { Link, useLocation } from "react-router-dom";
import Paper from "@mui/material/Paper";

const Search = styled("div")(({ theme }) => ({
  position: "relative",
  borderRadius: theme.shape.borderRadius,
  backgroundColor: alpha(theme.palette.common.white, 0.15),
  "&:hover": {
    backgroundColor: alpha(theme.palette.common.white, 0.25),
  },
  marginLeft: 0,
  width: "100%",
  [theme.breakpoints.up("sm")]: {
    marginLeft: theme.spacing(1),
    width: "auto",
  },
}));

const SearchIconWrapper = styled("div")(({ theme }) => ({
  padding: theme.spacing(0, 2),
  height: "100%",
  position: "absolute",
  pointerEvents: "none",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
}));

const StyledInputBase = styled(InputBase)(({ theme }) => ({
  color: "inherit",
  "& .MuiInputBase-input": {
    padding: theme.spacing(1, 1, 1, 0),
    // vertical padding + font size from searchIcon
    paddingLeft: `calc(1em + ${theme.spacing(4)})`,
    transition: theme.transitions.create("width"),
    width: "100%",
    [theme.breakpoints.up("sm")]: {
      width: "12ch",
      "&:focus": {
        width: "20ch",
      },
    },
  },
}));

function ResponsiveAppBar() {
  const [searchString, setSearchString] = useState("");
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();

    // Fetch the processed string from the Flask backend
    const fetchProcessedString = async () => {
      try {
        const response = await fetch("/prot_folder", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ searchString }),
        });

        const data = await response.json();
        if (data.prot_folder !== "") {
          // Navigate to the "results" page with the input string
          navigate(`/results/${encodeURIComponent(searchString)}`);
        } else {
          // Navigate to the "results" page with the input string
          navigate(`/notfound`);
        }
      } catch (error) {
        console.error("Error:", error);
      }
    };

    fetchProcessedString();
  };

  const location = useLocation();
  const currentPage = location.pathname;
  console.log(currentPage);

  return (
    <AppBar position="static">
      <Container maxWidth="xl">
        <div className="container">
          <Toolbar disableGutters>
            <Typography
              variant="h6"
              noWrap
              component={Link}
              to="/"
              sx={{
                mr: 2,
                display: { xs: "none", md: "flex" },
                fontFamily: "monospace",
                fontWeight: 700,
                letterSpacing: ".01rem",
                color: "inherit",
                textDecoration: "none",
              }}
            >
              BENDER DB
            </Typography>
            <Box sx={{ flexGrow: 1, display: { xs: "none", md: "flex" } }}>
              <Typography
                variant="body1"
                noWrap
                component={Link}
                to="/results/A4HXH5"
                sx={{
                  p: 2,
                  display: { xs: "none", md: "flex" },
                  fontFamily: "monospace",
                  fontSize: 14,
                  fontWeight: 300,
                  letterSpacing: ".01rem",

                  textDecoration: "none",
                  "&:hover": { color: "#1976d2", backgroundColor: "white" },
                  transition: "color 0.3s, background-color 0.3s", // Add transition
                  backgroundColor: currentPage.startsWith("/results/")
                    ? "white"
                    : "#1976d2",
                  color: currentPage.startsWith("/results/")
                    ? "#1976d2"
                    : "white",
                }}
              >
                EXAMPLE
              </Typography>
              <Typography
                variant="body1"
                noWrap
                component={Link}
                to="/datatable"
                sx={{
                  p: 2,
                  display: { xs: "none", md: "flex" },
                  fontFamily: "monospace",
                  fontSize: 14,
                  fontWeight: 300,
                  letterSpacing: ".01rem",
                  textDecoration: "none",
                  "&:hover": {
                    color: "#1976d2",
                    backgroundColor: "white",
                    transition: "color 0.3s, background-color 0.3s", // Add transition
                  },
                  backgroundColor:
                    currentPage === "/datatable" ? "white" : "#1976d2",
                  color: currentPage === "/datatable" ? "#1976d2" : "white",
                }}
              >
                AVAILABLE DATA
              </Typography>

              <Typography
                variant="body1"
                noWrap
                component={Link}
                to="/contact"
                sx={{
                  p: 2,
                  display: { xs: "none", md: "flex" },
                  fontFamily: "monospace",
                  fontWeight: 300,
                  fontSize: 14,
                  letterSpacing: ".01rem",

                  textDecoration: "none",
                  "&:hover": { color: "#1976d2", backgroundColor: "white" },
                  transition: "color 0.3s, background-color 0.3s", // Add transition
                  backgroundColor:
                    currentPage === "/contact" ? "white" : "#1976d2",
                  color: currentPage === "/contact" ? "#1976d2" : "white",
                }}
              >
                CONTACT
              </Typography>
              <Typography
                variant="body1"
                noWrap
                component={Link}
                to="/help"
                sx={{
                  p: 2,
                  display: { xs: "none", md: "flex" },
                  fontFamily: "monospace",
                  fontWeight: 300,
                  fontSize: 14,
                  letterSpacing: ".01rem",

                  textDecoration: "none",
                  "&:hover": { color: "#1976d2", backgroundColor: "white" },
                  transition: "color 0.3s, background-color 0.3s", // Add transition
                  backgroundColor:
                    currentPage === "/help" ? "white" : "#1976d2",
                  color: currentPage === "/help" ? "#1976d2" : "white",
                }}
              >
                HELP
              </Typography>
            </Box>
            <Box sx={{ flexGrow: 0 }}>
              <Paper component="form" onSubmit={handleSubmit}>
                <Search>
                  <SearchIconWrapper>
                    <SearchIcon />
                  </SearchIconWrapper>
                  <StyledInputBase
                    onChange={(e) =>
                      setSearchString(e.target.value.toUpperCase())
                    }
                    placeholder="Search..."
                    inputProps={{ "aria-label": "search" }}
                  />
                </Search>
              </Paper>
            </Box>
          </Toolbar>
        </div>
      </Container>
    </AppBar>
  );
}
export default ResponsiveAppBar;
