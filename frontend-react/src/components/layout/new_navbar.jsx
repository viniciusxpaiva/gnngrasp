import * as React from "react";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Toolbar from "@mui/material/Toolbar";
import IconButton from "@mui/material/IconButton";
import Typography from "@mui/material/Typography";
import Menu from "@mui/material/Menu";
import MenuIcon from "@mui/icons-material/Menu";
import Container from "@mui/material/Container";
import Button from "@mui/material/Button";
import MenuItem from "@mui/material/MenuItem";
import { Link, useLocation } from "react-router-dom";

//const pagesLinks = { 'Example': '/results/A4HXH5', 'Available data': '/datatable', 'Contact': '/contact', 'Help': '/help' };
const pagesLinks = {
  "Available data": "/datatable",
  Contact: "/contact",
  Help: "/help",
};

function ResponsiveAppBar() {
  const [anchorElNav, setAnchorElNav] = React.useState(null);

  const handleOpenNavMenu = (event) => {
    setAnchorElNav(event.currentTarget);
  };

  const handleCloseNavMenu = () => {
    setAnchorElNav(null);
  };
  const location = useLocation();
  const currentPage = location.pathname;

  return (
    <AppBar position="static">
      <div className="container">
        <Container maxWidth="xl">
          <Toolbar disableGutters sx={{ display: "flex" }}>
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
              BENDER DB |
            </Typography>
            <Typography
              variant="body2"
              sx={{ display: { xs: "none", md: "flex" } }}
            >
              PROTEIN BINDING SITE DATABASE
            </Typography>

            <Box sx={{ flexGrow: 1, display: { xs: "flex", md: "none" } }}>
              <IconButton
                size="large"
                aria-label="account of current user"
                aria-controls="menu-appbar"
                aria-haspopup="true"
                onClick={handleOpenNavMenu}
                color="inherit"
              >
                <MenuIcon />
              </IconButton>
              <Menu
                id="menu-appbar"
                anchorEl={anchorElNav}
                anchorOrigin={{
                  vertical: "bottom",
                  horizontal: "left",
                }}
                keepMounted
                transformOrigin={{
                  vertical: "top",
                  horizontal: "left",
                }}
                open={Boolean(anchorElNav)}
                onClose={handleCloseNavMenu}
                sx={{
                  display: { xs: "block", md: "none" },
                }}
              >
                {Object.entries(pagesLinks).map(([k, v]) => (
                  <MenuItem key={k} component={Link} to={v}>
                    <Typography textAlign="center">{k} </Typography>
                  </MenuItem>
                ))}
              </Menu>
            </Box>
            <Typography
              variant="h5"
              noWrap
              component={Link}
              to="/"
              sx={{
                mr: 2,
                display: { xs: "flex", md: "none" },
                flexGrow: 1,
                fontFamily: "monospace",
                fontWeight: 700,
                letterSpacing: ".1rem",
                color: "inherit",
                textDecoration: "none",
              }}
            >
              BENDER DB
            </Typography>
            <Box
              sx={{
                marginLeft: "auto",
                flexGrow: 0,
                display: { xs: "none", md: "flex" },
              }}
            >
              {Object.entries(pagesLinks).map(([k, v]) => (
                <Button
                  key={k}
                  component={Link}
                  to={v}
                  sx={{ my: 2, color: "white", display: "block" }}
                >
                  <Typography
                    variant="body2"
                    sx={{
                      textDecoration: "none",
                      fontWeight: currentPage === v ? "bold" : "normal",
                    }}
                  >
                    {k}
                  </Typography>
                </Button>
              ))}
            </Box>
          </Toolbar>
        </Container>
      </div>
    </AppBar>
  );
}
export default ResponsiveAppBar;
