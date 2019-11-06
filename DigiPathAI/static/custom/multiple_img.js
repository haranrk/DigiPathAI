$(document).ready(function() {
    $("#toggle-button").prop('checked', false); 
    var dzi_data = { dzi_data | default('{}') | safe };
    var viewer = new OpenSeadragon({
        id: "view",
        prefixUrl: "static/images/",
        timeout: 120000,
        animationTime: 0.5,
        blendTime: 0.1,
        constrainDuringPan: true,
        maxZoomPixelRatio: 2,
        minZoomLevel: 1,
        visibilityRatio: 1,
        zoomPerScroll: 2,
    });
    viewer.addHandler("open", function() {
        // To improve load times, ignore the lowest-resolution Deep Zoom
        // levels.  This is a hack: we can't configure the minLevel via
        // OpenSeadragon configuration options when the viewer is created
        // from DZI XML.
        viewer.source.minLevel = 8;
    });
    viewer.scalebar({
        xOffset: 10,
        yOffset: 10,
        barThickness: 3,
        color: '#555555',
        fontColor: '#333333',
        backgroundColor: 'rgba(255, 255, 255, 0.5)',
    });

    function open_slide(url, mpp) {
        var tile_source, mask_source;
        var mask_url = url.replace("slide", "mask");

        console.log(url);
        if (dzi_data[url]) {
            console.log("if");
            // DZI XML provided as template argument (deepzoom_tile.py)
            tile_source = new OpenSeadragon.DziTileSource(
                    OpenSeadragon.DziTileSource.prototype.configure(
                    OpenSeadragon.parseXml(dzi_data[url]), url));
            mask_source = new OpenSeadragon.DziTileSource(
                    OpenSeadragon.DziTileSource.prototype.configure(
                    OpenSeadragon.parseXml(dzi_data[mask_url]), mask_url));
        } else {
            console.log("else");
            // DZI XML fetched from server (deepzoom_server.py)
            tile_source = url;
            mask_source = mask_url
        }
        viewer.open(tile_source);
        viewer.addTiledImage({
            tileSource: mask_source,
            opacity: 0,
        });

        $("#toggle-btn").change(function(){
            tiledImage = viewer.world.getItemAt(1);
            if($(this).is(':checked')) {
                console.log("checked");
                tiledImage.setOpacity(0.5);
            } else {
                tiledImage.setOpacity(0);
            }
        });

        viewer.scalebar({
            pixelsPerMeter: mpp ? (1e6 / mpp) : 0,
        });
    }

    open_slide("{{ slide_url }}", parseFloat('{{ slide_mpp }}'));
//    $(".load-slide").click(function(ev) {
//        $(".current-slide").removeClass("current-slide");
//        $(this).parent().addClass("current-slide");
//        open_slide($(this).attr('data-url'),
//                parseFloat($(this).attr('data-mpp')));
//        ev.preventDefault();
//    });
});
