const imageUploadForm = document.querySelector('.image-upload-form');

const uploadImage = async (imageFile) => {
  const formData = new FormData();
  formData.append('file', imageFile);

  const inferResponse = await fetch('detect_license_plate', {
    method: 'POST',
    body: formData,
  });

  const inferBlob = await inferResponse.blob();
  return URL.createObjectURL(inferBlob);
};

imageUploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();

  const imageInput = document.getElementById('image-input');
  const imageOutput = document.querySelector('.image-output');

  try {
    imageOutput.textContent = 'Processing...';
    const resultUrl = await uploadImage(imageInput.files[0]);
    imageOutput.innerHTML = `<img src="${resultUrl}" alt="Processed Image"/>`;
  } catch (err) {
    console.error(err);
    imageOutput.textContent = 'An error occurred.';
  }
});
