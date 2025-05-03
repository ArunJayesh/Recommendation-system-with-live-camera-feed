const jsonServer = require('json-server');
const path = require('path');

// Create server
const server = jsonServer.create();

// Set default middlewares (logger, static, CORS, etc)
const middlewares = jsonServer.defaults();
server.use(middlewares);

// Point to your JSON file
const router = jsonServer.router(path.join(__dirname, 'db.json'));

// Use default router
server.use(router);

// Start server
const port = process.env.PORT || 3000;
server.listen(port, () => {
  console.log(`🚀 JSON‑Server running at http://localhost:${port}`);
});
