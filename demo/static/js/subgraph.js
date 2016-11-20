// 路径配置

require.config({
    paths: {
        echarts: '../static/js/echarts-2.2.7/build/dist',
        theme: '../static/js/echarts-2.2.7/src/theme'
    }
});
// 使用
require(['theme/mint'], function(theme) {
    mint = theme;
});

$(function(){
    $('#get-subgraph').bind('click', function(){
        $.getJSON('/get_subgraph', {
            mid: $("#subgraph-input").val(),
        }, function(data) {
            draw_graph(data.nodes, data.links);
        });
        return false;
    })
})

function draw_graph(nodes, links){
    require([
            'echarts',
            'echarts/chart/force',
        ],
        function (ec) {
            // 基于准备好的dom，初始化echarts图表
            var myChart = ec.init(document.getElementById('subgraph'), mint);

            option = {
                title : {
                    text: 'Freebase Subgraph',
//                    subtext: 'kbqa',
                    x:'right',
                    y:'bottom'
                },
                tooltip : {
                    trigger: 'item',
                    formatter: '{a} : {b}'
                },
                toolbox: {
                    show : true,
                    feature : {
                        restore : {show: true},
//                        magicType: {show: true, type: ['force', 'chord']},
//                        saveAsImage : {show: true}
                    }
                },
                legend: {
                    x: 'left',
                    data:['主题实体','匿名结点', '实体']
                },
                series : [
                    {
                        type:'force',
                        name : "Freebase",
                        ribbonType: false,
                        categories : [
                            {
                                name: '主题实体'
                            },
                            {
                                name: '匿名结点'
                            },
                            {
                                name:'实体'
                            }
                        ],
                        itemStyle: {
                            normal: {
                                label: {
                                    show: true,
                                    textStyle: {
                                        color: '#333'
                                    }
                                },
                                nodeStyle : {
                                    brushType : 'both',
                                    borderColor : 'rgba(255,215,0,0.4)',
                                    borderWidth : 1
                                },
                                linkStyle: {
                                    type: 'curve'
                                }
                            },
                            emphasis: {
                                label: {
                                    show: false
                                    // textStyle: null      // 默认使用全局文本样式，详见TEXTSTYLE
                                },
                                nodeStyle : {
                                    //r: 30
                                },
                                linkStyle : {}
                            }
                        },
                        useWorker: false,
                        minRadius : 15,
                        maxRadius : 25,
                        gravity: 1.1,
                        scaling: 1.1,
                        roam: 'move',
                        linkSymbol: 'arrow',
                        nodes: nodes,
                        links : links
                    }
                ]
            };
            var ecConfig = require('echarts/config');
            function focus(param) {
                var data = param.data;
                var links = option.series[0].links;
                var nodes = option.series[0].nodes;
                if (
                    data.source !== undefined
                    && data.target !== undefined
                ) { //点击的是边
                    var sourceNode = nodes.filter(function (n) {return n.name == data.source})[0];
                    var targetNode = nodes.filter(function (n) {return n.name == data.target})[0];
                    console.log("选中了边 " + sourceNode.name + ' -> ' + targetNode.name + ' (' + data.weight + ')');
                } else { // 点击的是点
                    console.log("选中了" + data.name + '(' + data.value + ')');
                }
            }
            myChart.on(ecConfig.EVENT.CLICK, focus)

            myChart.on(ecConfig.EVENT.FORCE_LAYOUT_END, function () {
                console.log(myChart.chart.force.getPosition());
            });


            // 为echarts对象加载数据
            myChart.setOption(option);

        }
    );
}
//draw_subgraph(nodes, links);