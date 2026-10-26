module.exports = {
  ...require('./kernels'),
  ...require('./info'),
  ...require('./monitor'),
  market: require('./apps/market'),
};
