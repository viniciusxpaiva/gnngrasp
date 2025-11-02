import React, { useEffect, useState } from "react";
import { extractCombinations, render } from '@upsetjs/bundle';

export default function UpsetPlot(props) {
    const [selection, setSelection] = useState(null);

    const { sets, combinations } = extractCombinations(props.data);

    function onClick(set) {
        if (set) {
            console.log(set);
            props.upsetOnClick(set);
            setSelection(set);
        }
    }

    useEffect(() => {
        rerender();
    }, [sets, combinations, selection]);

    function rerender() {
        try {
            const container = document.getElementById('upset-plot-container');
            const parentContainer = document.getElementById('upset-parent-container');
            const maxWidth = parentContainer.clientWidth; // Use the width of the parent container
            const width = maxWidth > 800 ? maxWidth : 800; // Ensure a minimum width of 800px
            const height = 300;

            const sortedCombinations = combinations.slice().sort((a, b) => b.cardinality - a.cardinality);

            const plotProps = { sets: sets, sortedCombinations, width, height, selection, onClick };
            render(container, plotProps);
        } catch (error) {
            console.error('Error in UpSetPlotComponent:', error);
        }
    }

    useEffect(() => {
        const handleResize = () => {
            rerender();
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, []);

    return (
        <div id="upset-parent-container" style={{ overflowX: 'auto', overflowY: 'hidden', width: '100%' }}>
            <div id="upset-plot-container" style={{ minWidth: '800px', width: '100%' }}></div>
        </div>
    );
}
