const fs = require('fs');
const path = require('path');

// Copy public directory to standalone
const publicSrc = path.join(__dirname, '..', 'public');
const publicDest = path.join(__dirname, '..', '.next', 'standalone', 'ui', 'src', 'ogx_ui', 'public');

if (fs.existsSync(publicSrc) && !fs.existsSync(publicDest)) {
  console.log('Copying public directory to standalone...');
  copyDir(publicSrc, publicDest);
}

// Copy .next/static to standalone
const staticSrc = path.join(__dirname, '..', '.next', 'static');
const staticDest = path.join(__dirname, '..', '.next', 'standalone', 'ui', 'src', 'ogx_ui', '.next', 'static');

if (fs.existsSync(staticSrc) && !fs.existsSync(staticDest)) {
  console.log('Copying .next/static to standalone...');
  copyDir(staticSrc, staticDest);
}

function copyDir(src, dest) {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  const files = fs.readdirSync(src);
  files.forEach((file) => {
    const srcFile = path.join(src, file);
    const destFile = path.join(dest, file);

    if (fs.statSync(srcFile).isDirectory()) {
      copyDir(srcFile, destFile);
    } else {
      fs.copyFileSync(srcFile, destFile);
    }
  });
}

console.log('Postbuild complete!');
