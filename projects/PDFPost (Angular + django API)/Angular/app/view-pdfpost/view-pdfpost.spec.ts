import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ViewPdfpost } from './view-pdfpost';

describe('ViewPdfpost', () => {
  let component: ViewPdfpost;
  let fixture: ComponentFixture<ViewPdfpost>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ViewPdfpost]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ViewPdfpost);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
