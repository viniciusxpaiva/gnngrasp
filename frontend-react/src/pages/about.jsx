import React from 'react';
import BaseLayout from '../components/layout/base';
import Stack from "@mui/material/Stack";
import Tabs from "@mui/material/Tabs";
import Tab from "@mui/material/Tab";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import PropTypes from "prop-types";
import Card from "@mui/material/Card";
import Divider from "@mui/material/Divider";

function a11yProps(index) {
  return {
    id: `simple-tab-${index}`,
    "aria-controls": `simple-tabpanel-${index}`,
  };
}

CustomTabPanel.propTypes = {
  children: PropTypes.node,
  index: PropTypes.number.isRequired,
  value: PropTypes.number.isRequired,
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
          <Typography component="div">{children}</Typography>
        </Box>
      )}
    </div>
  );
}

const About = () => {
  return (
    <BaseLayout>
      {/* Header */}
      <div className="container-fluid bg-light-dark text-white mt-0 py-4" id="about-header">
        <div className="row justify-content-center">
          <div className="col-md-12 text-center">
            <h6 className="display-6 text-light">About GNN-GRaSP</h6>
            <p className="text-light-50 mt-2 mb-0">
              A hierarchical graph-based predictor for protein–ligand binding sites
            </p>
          </div>
        </div>
      </div>

      {/* Main container */}
      <div className="container">
        <Box
          sx={{
            borderBottom: 1,
            borderColor: "divider",
            display: "flex",
            justifyContent: "center",
          }}
        >
          <Tabs
            aria-label="about tabs"
            variant="scrollable"
            scrollButtons="auto"
            value={0}
          >
            <Tab label={"About"} {...a11yProps(0)} />
          </Tabs>
        </Box>

        <CustomTabPanel value={0} index={0}>
          <Card variant="outlined">
            <Box sx={{ p: 2 }}>
              <div className="row mt-2 mb-2">
                <div className="col-12">

                  {/* -------- SUBSECTION 1 -------- */}
                  <Typography gutterBottom variant="h5" component="div">
                    CLARA: A Hierarchical Framework for Imbalanced Graph Learning
                  </Typography>

                  <Typography color="text.secondary" variant="body1" paragraph sx={{ textAlign: "justify" }}>
                    GNN-GRaSP is built upon <strong>CLARA</strong> (Coarse-to-fine 
                    Localized Adaptive Region Attention), a general framework designed to 
                    address <em>class imbalance in graph-structured data</em>. Many graph 
                    learning problems suffer from extremely skewed label distributions, 
                    where minority instances appear only in compact, localized regions of 
                    the network. Traditional methods—such as oversampling, loss reweighting, 
                    or global augmentation—treat imbalance as a uniform phenomenon and often 
                    fail when positive instances form small, meaningful clusters.
                  </Typography>

                  <Typography color="text.secondary" variant="body1" paragraph sx={{ textAlign: "justify" }}>
                    CLARA tackles this by breaking node classification into two coordinated 
                    stages. First, a <strong>subgraph-level classifier</strong> identifies 
                    regions of the graph likely to contain positive instances, filtering out 
                    irrelevant areas dominated by the majority class. Then, a 
                    <strong> fine-grained node classifier</strong> operates only on those 
                    selected regions. Both stages are implemented using Graph Attention 
                    Networks (GATs), allowing the model to emphasize informative neighbors 
                    during message passing. This hierarchical approach improves sensitivity 
                    to rare classes while preserving efficiency and robustness across diverse 
                    graph scenarios.
                  </Typography>


                  {/* -------- SUBSECTION 2 -------- */}
                  <Typography gutterBottom variant="h5" component="div" sx={{ mt: 4 }}>
                    Graph Modeling for Protein Structures
                  </Typography>

                  <Typography color="text.secondary" variant="body1" paragraph sx={{ textAlign: "justify" }}>
                    In our application, proteins are represented as <strong>residue-level 
                    graphs</strong>, where nodes correspond to amino acids and edges encode 
                    spatial or interaction-based relationships such as hydrogen bonds, 
                    hydrophobic contacts, and aromatic interactions. Instead of hand-crafted 
                    biochemical descriptors, each node is characterized by 
                    <strong> ESM-2 embeddings</strong>, a powerful representation learned by a 
                    large protein language model that captures evolutionary and structural 
                    information directly from amino acid sequences.
                  </Typography>

                  <Typography color="text.secondary" variant="body1" paragraph sx={{ textAlign: "justify" }}>
                    A crucial step in CLARA is the decomposition of proteins into localized 
                    <strong> subgraphs</strong>. GNN-GRaSP supports both randomized coverage 
                    strategies and domain-informed heuristics. Of particular relevance is the 
                    <strong> SASA-based heuristic</strong>, where residues with high 
                    solvent-accessible surface area are chosen as subgraph seeds. Because 
                    ligand-binding residues almost always occur on the protein surface, this 
                    heuristic significantly improves the model’s focus on functionally 
                    relevant regions.
                  </Typography>


                  {/* -------- SUBSECTION 3 -------- */}
                  <Typography gutterBottom variant="h5" component="div" sx={{ mt: 4 }}>
                    GNN-GRaSP: Applying CLARA to Binding Site Prediction
                  </Typography>

                  <Typography color="text.secondary" variant="body1" paragraph sx={{ textAlign: "justify" }}>
                    GNN-GRaSP represents a specialized adaptation of CLARA for 
                    <strong> protein–ligand binding site prediction</strong>. Binding residues 
                    form small, highly localized clusters on the protein surface, making this 
                    task inherently imbalanced. By first identifying subgraphs likely to 
                    contain binding residues and then refining predictions within those 
                    regions, GNN-GRaSP captures the spatial organization of functional 
                    residues more effectively than whole-protein models.
                  </Typography>

                  <Typography color="text.secondary" variant="body1" paragraph sx={{ textAlign: "justify" }}>
                    When evaluated on the COACH100 benchmark, our hierarchical variant 
                    leveraging SASA-based subgraph generation achieved a 
                    <strong> Matthews Correlation Coefficient of 0.674</strong> and an 
                    <strong> F1-score of 68.6%</strong>, surpassing state-of-the-art binding 
                    site predictors such as PUResNet v2.0 and GRaSP, as well as specialized 
                    graph-imbalance solutions like NodeImport, GraphSHA, and BAT. These 
                    results highlight the advantages of modeling binding sites as clustered 
                    subregions rather than isolated nodes, and demonstrate the value of 
                    incorporating domain-specific heuristics into graph-based learning.
                  </Typography>

                  <Typography color="text.secondary" variant="body1" paragraph sx={{ textAlign: "justify" }}>
                    Overall, GNN-GRaSP combines CLARA’s hierarchical architecture with rich 
                    protein representations and biologically guided subgraph generation, 
                    offering a powerful and extensible framework for studying ligand-binding 
                    mechanisms and other structure-based molecular tasks.
                  </Typography>

                </div>
              </div>
            </Box>
            <Divider />
          </Card>
        </CustomTabPanel>
      </div>
    </BaseLayout>
  );
};

export default About;
