import { Component } from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-create-pdfpost',
  imports: [],
  templateUrl: './create-pdfpost.html',
  styleUrl: './create-pdfpost.scss',
})
export class CreatePdfpost {
  constructor(private router: Router) {}

  goHome() {
    this.router.navigate(['/']);
  }
}
