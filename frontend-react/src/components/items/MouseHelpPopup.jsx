import * as React from "react";
import Box from "@mui/material/Box";
import IconButton from "@mui/material/IconButton";
import MouseIcon from "@mui/icons-material/Mouse";
import Typography from "@mui/material/Typography";
import Divider from "@mui/material/Divider";
import Dialog from "@mui/material/Dialog";
import DialogActions from "@mui/material/DialogActions";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Button from "@mui/material/Button";
import CloseIcon from "@mui/icons-material/Close";
import Tooltip, { tooltipClasses } from '@mui/material/Tooltip';
import { styled } from '@mui/system';


const NoMaxWidthTooltip = styled(({ className, ...props }) => (
  <Tooltip {...props} classes={{ popper: className }} arrow />
))({
  [`& .${tooltipClasses.tooltip}`]: {
    maxWidth: 'none',
  },
});

export default function MouseHelpPopup(props) {
  const [open, setOpen] = React.useState(false);

  const handleClickOpen = () => {
    setOpen(true);
  };

  const handleClose = (event, reason) => {
    if (reason !== "backdropClick") {
      setOpen(false);
    }
  };

  return (
    <div>
      <NoMaxWidthTooltip title="Mouse controls">

        <IconButton onClick={handleClickOpen}>
          <MouseIcon htmlColor={props.bgroundColor === "white" ? "" : "white"} />
        </IconButton>

      </NoMaxWidthTooltip>

      <Dialog disableEscapeKeyDown open={open} onClose={handleClose}>
        <DialogTitle>Mouse controls</DialogTitle>
        <DialogContent>
          <Typography color="text.secondary" variant="body2">
            Following mouse commands can be used at molecular visualization
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
          <Box>
            <div class="container">
              <div class="row">
                <div class="col-md-2">
                  <img
                    src="../img/mouseleftclick.jpg"
                    alt=""
                    style={{ width: "50px", height: "50px" }}
                  />
                </div>
                <div class="col-md-10 p-0 d-flex align-items-center">
                  <Typography color="text.secondary" variant="body2">
                    <b>Left button click</b>: pick atom.
                  </Typography>
                </div>
              </div>
              <div class="row mt-2">
                <div class="col-md-2">
                  <img
                    src="../img/mousescroll.jpg"
                    alt=""
                    style={{ width: "50px", height: "50px" }}
                  />
                </div>
                <div class="col-md-10 p-0 d-flex align-items-center">
                  <Typography color="text.secondary" variant="body2">
                    <b>Middle button scroll</b>: zoom camera.
                  </Typography>
                </div>
              </div>

              <div class="row mt-2">
                <div class="col-md-2">
                  <img
                    src="../img/mousemiddleclick.jpg"
                    alt=""
                    style={{ width: "50px", height: "50px" }}
                  />
                </div>
                <div class="col-md-10 p-0 d-flex align-items-center">
                  <Typography color="text.secondary" variant="body2">
                    <b>Middle button click</b>: center camera on atom.
                  </Typography>
                </div>
              </div>

              <div class="row mt-2">
                <div class="col-md-2">
                  <img
                    src="../img/mousemiddlehold.jpg"
                    alt=""
                    style={{ width: "50px", height: "50px" }}
                  />
                </div>
                <div class="col-md-10 p-0 d-flex align-items-center">
                  <Typography color="text.secondary" variant="body2">
                    <b>Middle button hold and move</b>: zoom camera in and out.
                  </Typography>
                </div>
              </div>

              <div class="row mt-2">
                <div class="col-md-2">
                  <img
                    src="../img/mouselefthold.jpg"
                    alt=""
                    style={{ width: "50px", height: "50px" }}
                  />
                </div>
                <div class="col-md-10 p-0 d-flex align-items-center">
                  <Typography color="text.secondary" variant="body2">
                    <b>Left button hold and move</b>: rotate camera around
                    center.
                  </Typography>
                </div>
              </div>

              <div class="row mt-2 mb-2">
                <div class="col-md-2">
                  <img
                    src="../img/mouserighthold.jpg"
                    alt=""
                    style={{ width: "50px", height: "50px" }}
                  />
                </div>
                <div class="col-md-10 p-0 d-flex align-items-center">
                  <Typography color="text.secondary" variant="body2">
                    <b>Right button hold and move</b>: translate camera in
                    screen plane.
                  </Typography>
                </div>
              </div>
            </div>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose}>Close</Button>
        </DialogActions>
      </Dialog>
    </div>
  );
}
