import fire
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


class Upload:
    gauth = GoogleAuth()
    gauth.CommandLineAuth()
    drive = GoogleDrive(gauth)

    def file(self, file):
        file = self.drive.CreateFile({f'title': file})
        file.SetContentFile(file)
        file.Upload()  # Upload the file.
        print('title: %s, id: %s' % (file['title'], file['id']))

    def models(self):
        quant_noise_probs = [0, 0.25, 0.5, 0.75, 1]
        for q in quant_noise_probs:
            self.file(f'model_{q}.pt')


if __name__ == '__main__':
    fire.Fire(Upload)
