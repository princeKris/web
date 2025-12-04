import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ViweProfile } from './viwe-profile';

describe('ViweProfile', () => {
  let component: ViweProfile;
  let fixture: ComponentFixture<ViweProfile>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ViweProfile]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ViweProfile);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
