import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CreatePdfpost } from './create-pdfpost';

describe('CreatePdfpost', () => {
  let component: CreatePdfpost;
  let fixture: ComponentFixture<CreatePdfpost>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [CreatePdfpost]
    })
    .compileComponents();

    fixture = TestBed.createComponent(CreatePdfpost);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
